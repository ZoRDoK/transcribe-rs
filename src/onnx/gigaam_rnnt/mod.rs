use ort::inputs;
use ort::session::Session;
use ort::value::TensorRef;
use std::path::Path;

use super::session;
use super::Quantization;
use crate::decode::tokens::load_vocab;
use crate::decode::sentencepiece_to_text;
use crate::features::{compute_mel, MelConfig, WindowType};
use crate::TranscribeError;
use crate::{ModelCapabilities, SpeechModel, TranscribeOptions, TranscriptionResult};

const CAPABILITIES: ModelCapabilities = ModelCapabilities {
    name: "GigaAM RNNT",
    engine_id: "gigaam_rnnt",
    sample_rate: 16000,
    languages: &["ru"],
    supports_timestamps: false,
    supports_translation: false,
    supports_streaming: false,
};

const BLANK_ID: i64 = 1024;

/// Per-model inference parameters for GigaAM RNNT.
#[derive(Debug, Clone, Default)]
pub struct GigaAMRNNTParams {
    /// Language hint (unused, GigaAM is Russian-only).
    pub language: Option<String>,
}

pub struct GigaAMRNNTModel {
    encoder: Session,
    decoder: Session,
    joiner: Session,
    mel_config: MelConfig,
    vocab: Vec<String>,
}

impl GigaAMRNNTModel {
    pub fn load(model_dir: &Path, quantization: &Quantization) -> Result<Self, TranscribeError> {
        let suffix = match quantization {
            Quantization::Int8 => ".int8.onnx",
            Quantization::FP16 => ".fp16.onnx",
            Quantization::Int4 => ".int4.onnx",
            Quantization::FP32 => ".onnx",
        };

        let encoder_path = model_dir.join(&format!("encoder{}", suffix));
        let decoder_path = model_dir.join(&format!("decoder{}", suffix));
        let joiner_path = model_dir.join(&format!("joiner{}", suffix));
        let vocab_path = model_dir.join("vocab.txt");

        // Fallback to .onnx if quantized not found
        let encoder_path = if encoder_path.exists() { encoder_path }
                          else { model_dir.join("encoder.onnx") };
        let decoder_path = if decoder_path.exists() { decoder_path }
                          else { model_dir.join("decoder.onnx") };
        let joiner_path = if joiner_path.exists() { joiner_path }
                         else { model_dir.join("joiner.onnx") };

        if !encoder_path.exists() {
            return Err(TranscribeError::ModelNotFound(encoder_path));
        }
        if !decoder_path.exists() {
            return Err(TranscribeError::ModelNotFound(decoder_path));
        }
        if !joiner_path.exists() {
            return Err(TranscribeError::ModelNotFound(joiner_path));
        }
        if !vocab_path.exists() {
            return Err(TranscribeError::ModelNotFound(vocab_path));
        }

        log::info!("Loading GigaAM RNNT encoder from {:?}...", encoder_path);
        log::info!("Loading GigaAM RNNT decoder from {:?}...", decoder_path);
        log::info!("Loading GigaAM RNNT joiner from {:?}...", joiner_path);

        let encoder = session::create_session(&encoder_path)?;
        let decoder = session::create_session(&decoder_path)?;
        let joiner = session::create_session(&joiner_path)?;

        let (vocab, _) = load_vocab(&vocab_path)?;

        log::info!(
            "Loaded GigaAM RNNT vocabulary with {} tokens (blank={})",
            vocab.len(),
            BLANK_ID
        );

        // GigaAM v3 E2E RNNT feature extraction config
        // Source: sherpa-onnx IsGigaAM() override + GigaAM official repo
        // https://github.com/salute-developers/GigaAM/blob/main/gigaam/preprocess.py#L68
        let mel_config = MelConfig {
            sample_rate: 16000,
            num_mels: 64,
            n_fft: 400,        // was 320 (incorrect); GigaAM uses 400
            hop_length: 160,
            window: WindowType::Hann,
            f_min: 0.0,
            f_max: Some(8000.0),
            pre_emphasis: None,  // GigaAM uses pre_emph_coeff=0
            snip_edges: false,
            normalize_samples: true,
        };

        Ok(Self {
            encoder,
            decoder,
            joiner,
            mel_config,
            vocab,
        })
    }

    /// Transcribe with model-specific parameters.
    pub fn transcribe_with(
        &mut self,
        samples: &[f32],
        _params: &GigaAMRNNTParams,
    ) -> Result<TranscriptionResult, TranscribeError> {
        self.infer(samples)
    }

    fn infer(&mut self, samples: &[f32]) -> Result<TranscriptionResult, TranscribeError> {
        if samples.len() < self.mel_config.n_fft {
            return Ok(TranscriptionResult {
                text: String::new(),
                segments: None,
            });
        }

        // 1. Compute mel spectrogram [frames, mels]
        let mel = compute_mel(samples, &self.mel_config);
        let time_steps = mel.shape()[0];

        log::debug!(
            "Mel spectrogram shape: [{}, {}]",
            mel.shape()[0],
            mel.shape()[1]
        );

        // 2. Prepare input tensors: features [1, n_mels, time], feature_lengths [1]
        let features = mel.t().insert_axis(ndarray::Axis(0));
        let features = features.as_standard_layout().into_owned();
        let features_dyn = features.into_dyn();
        let feature_lengths = ndarray::arr1(&[time_steps as i64]).into_dyn();

        // 3. Run encoder forward pass
        let t_features = TensorRef::from_array_view(features_dyn.view())?;
        let t_lengths = TensorRef::from_array_view(feature_lengths.view())?;
        let inputs = inputs! {
            "audio_signal" => t_features,
            "length" => t_lengths,
        };
        let outputs = self.encoder.run(inputs)?;

        // 4. Extract encoder output [1, 768, T']
        let encoded = outputs[0].try_extract_array::<f32>()?;
        let encoded = encoded.to_owned().into_dimensionality::<ndarray::Ix3>()?;

        log::debug!("Encoder output shape: {:?}", encoded.shape());

        // 5. RNNT greedy decode
        let max_time = encoded.shape()[2] as usize;

        // Decoder LSTM states (persist across entire utterance)
        let mut h = ndarray::Array3::<f32>::zeros((1, 1, 320)).as_standard_layout().into_owned();
        let mut c = ndarray::Array3::<f32>::zeros((1, 1, 320)).as_standard_layout().into_owned();

        let mut tokens: Vec<i64> = Vec::new();
        let mut prev_token_id = BLANK_ID;
        let vocab_size = self.vocab.len();

        for t in 0..max_time {
            // Get encoder output at time t: [1, 768, 1]
            let enc_t = encoded.slice(ndarray::s![.., .., t..t+1]).to_owned();

            for _step in 0..100 {
                let t_h = TensorRef::from_array_view(h.view())?;
                let t_c = TensorRef::from_array_view(c.view())?;
                let prev_token_arr = ndarray::arr2(&[[prev_token_id]]);
                let t_prev = TensorRef::from_array_view(prev_token_arr.view())?;

                let dec_outputs = self.decoder.run(inputs! {
                    "x" => t_prev,
                    "h.1" => t_h,
                    "c.1" => t_c,
                })?;

                // dec_outputs[0] = decoder embedding [1, 1, 320]
                let dec_emb = dec_outputs[0].try_extract_array::<f32>()?
                    .to_owned()
                    .into_dimensionality::<ndarray::Ix3>()?;

                // [1, 1, 320] -> [1, 320, 1] (batch, hidden, time)
                let dec_emb_t = dec_emb.view()
                    .permuted_axes([0, 2, 1])
                    .as_standard_layout()
                    .into_owned();

                // Update LSTM states from decoder output
                if let Ok(h_new) = dec_outputs[1].try_extract_array::<f32>() {
                    h = h_new.to_owned().into_dimensionality::<ndarray::Ix3>()?.as_standard_layout().into_owned();
                }
                if let Ok(c_new) = dec_outputs[2].try_extract_array::<f32>() {
                    c = c_new.to_owned().into_dimensionality::<ndarray::Ix3>()?.as_standard_layout().into_owned();
                }

                // Joiner: combine encoder and decoder outputs
                let t_enc = TensorRef::from_array_view(enc_t.view())?;
                let t_dec = TensorRef::from_array_view(dec_emb_t.view())?;

                let joiner_out = self.joiner.run(inputs! {
                    "enc" => t_enc,
                    "dec" => t_dec,
                })?;

                let logits = joiner_out[0].try_extract_array::<f32>()?
                    .to_owned()
                    .into_dimensionality::<ndarray::Ix4>()?;

                // logits shape: [1, 1, 1, vocab_size], take argmax
                let pred_token = (0..vocab_size)
                    .max_by(|&a, &b| logits[[0, 0, 0, a]].partial_cmp(&logits[[0, 0, 0, b]]).unwrap())
                    .unwrap_or(BLANK_ID as usize);

                if pred_token == BLANK_ID as usize {
                    break;
                }

                tokens.push(pred_token as i64);
                prev_token_id = pred_token as i64;
            }
        }

        log::debug!("Decoded {} raw tokens", tokens.len());

        // 6. Convert token IDs to text
        let text_tokens: Vec<&str> = tokens.iter()
            .filter_map(|&id| {
                let idx = id as usize;
                if idx < self.vocab.len() {
                    let token = self.vocab[idx].as_str();
                    if token == "<unk>" || token == "<blk>" {
                        None
                    } else {
                        Some(token)
                    }
                } else {
                    None
                }
            })
            .collect();

        log::debug!("Text tokens: {:?}", &text_tokens[..text_tokens.len().min(20)]);

        let text = sentencepiece_to_text(&text_tokens);

        Ok(TranscriptionResult {
            text,
            segments: None,
        })
    }
}

impl SpeechModel for GigaAMRNNTModel {
    fn capabilities(&self) -> ModelCapabilities {
        CAPABILITIES
    }

    fn transcribe_raw(
        &mut self,
        samples: &[f32],
        _options: &TranscribeOptions,
    ) -> Result<TranscriptionResult, TranscribeError> {
        self.infer(samples)
    }
}

use std::{io::Cursor, sync::OnceLock};

use image::{DynamicImage, ImageBuffer, ImageFormat, Rgba, imageops::FilterType};
use tract_onnx::prelude::*;

use crate::{Error, Result};

pub const MATTE_KEY_HEX: &str = "#FF00FF";

const MODEL_EDGE: u32 = 512;
const KEY: [u8; 3] = [255, 0, 255];
const TRANSPARENT_ALPHA: u8 = 3;

static MODEL: OnceLock<std::result::Result<TypedRunnableModel<TypedModel>, String>> =
    OnceLock::new();

pub fn prompt_for_foreground_matte(prompt: &str) -> String {
    format!(
        "{}\n\nTransparency preparation requirements:\n- Render the requested foreground subject isolated, entirely inside the frame, and separated from every image edge.\n- Use a perfectly flat, uniform RGB #FF00FF background.\n- Do not add scenery, gradients, texture, shadows, reflections, glow, or color spill to the background.",
        prompt.trim_end()
    )
}

fn build_model() -> Result<TypedRunnableModel<TypedModel>> {
    let mut reader = Cursor::new(include_bytes!("../assets/modnet.onnx"));
    tract_onnx::onnx()
        .model_for_read(&mut reader)
        .map_err(|error| {
            Error::State(format!(
                "could not load the foreground matting model: {error}"
            ))
        })?
        .with_input_fact(
            0,
            InferenceFact::dt_shape(
                f32::datum_type(),
                tvec!(1, 3, MODEL_EDGE as usize, MODEL_EDGE as usize),
            ),
        )
        .map_err(|error| {
            Error::State(format!(
                "could not configure the foreground matting model: {error}"
            ))
        })?
        .into_optimized()
        .map_err(|error| {
            Error::State(format!(
                "could not optimize the foreground matting model: {error}"
            ))
        })?
        .into_runnable()
        .map_err(|error| {
            Error::State(format!(
                "could not initialize the foreground matting model: {error}"
            ))
        })
}

fn model() -> Result<&'static TypedRunnableModel<TypedModel>> {
    MODEL
        .get_or_init(|| build_model().map_err(|error| error.to_string()))
        .as_ref()
        .map_err(|error| Error::State(error.clone()))
}

fn input_tensor(image: &image::RgbaImage) -> Result<Tensor> {
    let resized = image::imageops::resize(image, MODEL_EDGE, MODEL_EDGE, FilterType::Triangle);
    let mut values = Vec::with_capacity((MODEL_EDGE * MODEL_EDGE * 3) as usize);
    for channel in 0..3 {
        values.extend(
            resized
                .pixels()
                .map(|pixel| f32::from(pixel[channel]) / 255.0),
        );
    }
    Tensor::from_shape(&[1, 3, MODEL_EDGE as usize, MODEL_EDGE as usize], &values)
        .map_err(|error| Error::State(format!("could not prepare the foreground matte: {error}")))
}

fn alpha_mask(image: &image::RgbaImage) -> Result<ImageBuffer<image::Luma<u8>, Vec<u8>>> {
    let output = model()?
        .run(tvec!(input_tensor(image)?.into_tvalue()))
        .map_err(|error| Error::State(format!("foreground matting failed: {error}")))?;
    let alpha = output
        .first()
        .ok_or_else(|| Error::State("foreground matting model returned no alpha mask".into()))?
        .to_array_view::<f32>()
        .map_err(|error| {
            Error::State(format!(
                "foreground matting returned an invalid alpha mask: {error}"
            ))
        })?;
    if alpha.shape() != [1, 1, MODEL_EDGE as usize, MODEL_EDGE as usize] {
        return Err(Error::State(format!(
            "foreground matting returned an unexpected alpha-mask shape: {:?}",
            alpha.shape()
        )));
    }
    let mut mask = Vec::with_capacity((MODEL_EDGE * MODEL_EDGE) as usize);
    for y in 0..MODEL_EDGE as usize {
        for x in 0..MODEL_EDGE as usize {
            mask.push((alpha[[0, 0, y, x]].clamp(0.0, 1.0) * 255.0).round() as u8);
        }
    }
    ImageBuffer::from_vec(MODEL_EDGE, MODEL_EDGE, mask)
        .ok_or_else(|| Error::State("could not construct the foreground alpha mask".into()))
}

fn decontaminate(channel: u8, key: u8, alpha: f32) -> u8 {
    ((f32::from(channel) - (1.0 - alpha) * f32::from(key)) / alpha)
        .round()
        .clamp(0.0, 255.0) as u8
}

pub fn remove_background_with_foreground_matte(content: &[u8]) -> Result<Vec<u8>> {
    let mut image = image::load_from_memory(content)?.to_rgba8();
    let (width, height) = image.dimensions();
    let alpha =
        image::imageops::resize(&alpha_mask(&image)?, width, height, FilterType::CatmullRom);
    for (pixel, matte) in image.pixels_mut().zip(alpha.pixels()) {
        let alpha = (f32::from(matte[0]) / 255.0) * (f32::from(pixel[3]) / 255.0);
        let alpha_u8 = (alpha * 255.0).round() as u8;
        if alpha_u8 <= TRANSPARENT_ALPHA {
            *pixel = Rgba([0, 0, 0, 0]);
        } else if alpha_u8 < u8::MAX {
            *pixel = Rgba([
                decontaminate(pixel[0], KEY[0], alpha),
                decontaminate(pixel[1], KEY[1], alpha),
                decontaminate(pixel[2], KEY[2], alpha),
                alpha_u8,
            ]);
        } else {
            pixel[3] = alpha_u8;
        }
    }
    let mut output = Cursor::new(Vec::new());
    DynamicImage::ImageRgba8(image).write_to(&mut output, ImageFormat::Png)?;
    Ok(output.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn foreground_matte_prompt_requests_an_isolated_subject() {
        let prompt = prompt_for_foreground_matte("Nathan in green robes");
        assert!(prompt.contains("#FF00FF"));
        assert!(prompt.contains("separated from every image edge"));
    }

    #[test]
    fn decontamination_removes_magenta_spill_from_a_semtransparent_pixel() {
        assert_eq!(decontaminate(191, 255, 0.5), 127);
        assert_eq!(decontaminate(64, 0, 0.5), 128);
    }

    #[test]
    fn embedded_model_produces_a_png_alpha_mask() {
        let source = ImageBuffer::from_pixel(8, 8, Rgba([255, 0, 255, 255]));
        let mut encoded = Cursor::new(Vec::new());
        DynamicImage::ImageRgba8(source)
            .write_to(&mut encoded, ImageFormat::Png)
            .unwrap();
        let result = remove_background_with_foreground_matte(&encoded.into_inner()).unwrap();
        let result = image::load_from_memory(&result).unwrap().to_rgba8();
        assert_eq!(result.dimensions(), (8, 8));
    }
}

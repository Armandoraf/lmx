use std::io::Cursor;

use image::{DynamicImage, ImageFormat, Rgba};

use crate::Result;

pub const WHITE_KEY_HEX: &str = "#FFFFFF";

const KEY: [u8; 3] = [255, 255, 255];
const KEY_DISTANCE: u8 = 8;

pub fn prompt_for_white_key(prompt: &str) -> String {
    format!(
        "{}\n\nTransparency preparation requirements:\n- Render the requested character or sprite isolated, entirely inside the frame, and separated from every image edge.\n- Use a perfectly flat, uniform RGB #FFFFFF background.\n- Do not add scenery, gradients, texture, shadows, reflections, glow, or color spill to the background.",
        prompt.trim_end()
    )
}

fn is_key_candidate(pixel: &Rgba<u8>) -> bool {
    KEY.iter()
        .zip(pixel.0.iter())
        .all(|(key, channel)| key.abs_diff(*channel) <= KEY_DISTANCE)
}

fn enqueue_key_candidate(
    image: &image::RgbaImage,
    width: u32,
    background: &mut [bool],
    pending: &mut Vec<(u32, u32)>,
    x: u32,
    y: u32,
) {
    let index = (y * width + x) as usize;
    if !background[index] && is_key_candidate(image.get_pixel(x, y)) {
        background[index] = true;
        pending.push((x, y));
    }
}

fn border_connected_key_mask(image: &image::RgbaImage) -> Vec<bool> {
    let (width, height) = image.dimensions();
    let mut background = vec![false; (width * height) as usize];
    let mut pending = Vec::new();

    for x in 0..width {
        enqueue_key_candidate(image, width, &mut background, &mut pending, x, 0);
        if height > 1 {
            enqueue_key_candidate(image, width, &mut background, &mut pending, x, height - 1);
        }
    }
    for y in 1..height.saturating_sub(1) {
        enqueue_key_candidate(image, width, &mut background, &mut pending, 0, y);
        if width > 1 {
            enqueue_key_candidate(image, width, &mut background, &mut pending, width - 1, y);
        }
    }

    while let Some((x, y)) = pending.pop() {
        for (next_x, next_y) in [
            (x.checked_sub(1), Some(y)),
            (x.checked_add(1).filter(|next| *next < width), Some(y)),
            (Some(x), y.checked_sub(1)),
            (Some(x), y.checked_add(1).filter(|next| *next < height)),
        ] {
            if let (Some(next_x), Some(next_y)) = (next_x, next_y) {
                enqueue_key_candidate(image, width, &mut background, &mut pending, next_x, next_y);
            }
        }
    }
    background
}

fn decontaminate(channel: u8, key: u8, alpha: f32) -> u8 {
    ((f32::from(channel) - (1.0 - alpha) * f32::from(key)) / alpha)
        .round()
        .clamp(0.0, 255.0) as u8
}

fn touches_background(mask: &[bool], width: u32, height: u32, x: u32, y: u32) -> bool {
    [
        (x.checked_sub(1), Some(y)),
        (x.checked_add(1).filter(|next| *next < width), Some(y)),
        (Some(x), y.checked_sub(1)),
        (Some(x), y.checked_add(1).filter(|next| *next < height)),
    ]
    .into_iter()
    .any(|(next_x, next_y)| {
        next_x
            .zip(next_y)
            .is_some_and(|(next_x, next_y)| mask[(next_y * width + next_x) as usize])
    })
}

fn edge_alpha(pixel: &Rgba<u8>) -> f32 {
    KEY.iter()
        .zip(pixel.0.iter())
        .map(|(key, channel)| f32::from(key.abs_diff(*channel)) / 255.0)
        .fold(0.0, f32::max)
}

pub fn remove_white_key_background(content: &[u8]) -> Result<Vec<u8>> {
    let mut image = image::load_from_memory(content)?.to_rgba8();
    let (width, height) = image.dimensions();
    let background = border_connected_key_mask(&image);
    for (x, y, pixel) in image.enumerate_pixels_mut() {
        if background[(y * width + x) as usize] {
            *pixel = Rgba([0, 0, 0, 0]);
            continue;
        }
        if touches_background(&background, width, height, x, y) {
            let alpha = edge_alpha(pixel);
            if alpha > 0.0 && alpha < 1.0 {
                *pixel = Rgba([
                    decontaminate(pixel[0], KEY[0], alpha),
                    decontaminate(pixel[1], KEY[1], alpha),
                    decontaminate(pixel[2], KEY[2], alpha),
                    (alpha * 255.0).round() as u8,
                ]);
            }
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
    fn white_key_prompt_requests_an_isolated_sprite() {
        let prompt = prompt_for_white_key("Nathan in green robes");
        assert!(prompt.contains(WHITE_KEY_HEX));
        assert!(prompt.contains("character or sprite"));
        assert!(prompt.contains("separated from every image edge"));
    }

    #[test]
    fn removes_only_border_connected_white_background() {
        let mut source = image::RgbaImage::from_pixel(5, 5, Rgba([255, 255, 255, 255]));
        source.put_pixel(2, 2, Rgba([255, 255, 255, 255]));
        source.put_pixel(1, 2, Rgba([200, 0, 0, 255]));
        source.put_pixel(2, 1, Rgba([200, 0, 0, 255]));
        source.put_pixel(3, 2, Rgba([200, 0, 0, 255]));
        source.put_pixel(2, 3, Rgba([200, 0, 0, 255]));
        let mut encoded = Cursor::new(Vec::new());
        DynamicImage::ImageRgba8(source)
            .write_to(&mut encoded, ImageFormat::Png)
            .unwrap();

        let result = remove_white_key_background(&encoded.into_inner()).unwrap();
        let result = image::load_from_memory(&result).unwrap().to_rgba8();
        assert_eq!(result.get_pixel(0, 0)[3], 0);
        assert_eq!(result.get_pixel(2, 2), &Rgba([255, 255, 255, 255]));
        assert_eq!(result.get_pixel(1, 2), &Rgba([200, 0, 0, 255]));
    }

    #[test]
    fn decontaminates_a_semtransparent_edge_against_white() {
        assert_eq!(decontaminate(255, 255, 0.5), 255);
        assert_eq!(decontaminate(128, 255, 0.5), 1);
    }
}

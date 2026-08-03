use std::collections::VecDeque;
use std::io::Cursor;

use image::{DynamicImage, ImageFormat, Rgba};

use crate::Result;

pub const CHROMA_KEY_HEX: &str = "#FF00FF";
const KEY: [u8; 3] = [255, 0, 255];
const INNER_DISTANCE: f64 = 18.0;
const OUTER_DISTANCE: f64 = 110.0;

pub fn prompt_for_chroma_key(prompt: &str) -> String {
    format!(
        "{}\n\nTransparency preparation requirements:\n- Render the requested foreground subject isolated against a perfectly flat,\n  uniform chroma-key background of exact RGB #FF00FF.\n- Fill the entire background with only #FF00FF.\n- Do not add scenery, gradients, texture, shadows, reflections, glow, or color\n  spill to the background.\n- Keep the complete foreground subject inside the frame and separated from\n  every image edge.",
        prompt.trim_end()
    )
}

fn distance(pixel: Rgba<u8>) -> f64 {
    ((f64::from(pixel[0]) - f64::from(KEY[0])).powi(2)
        + (f64::from(pixel[1]) - f64::from(KEY[1])).powi(2)
        + (f64::from(pixel[2]) - f64::from(KEY[2])).powi(2))
    .sqrt()
}

fn recover(channel: u8, key: u8, alpha: f64) -> u8 {
    if alpha <= 0.0 {
        return 0;
    }
    ((f64::from(channel) - (1.0 - alpha) * f64::from(key)) / alpha)
        .round()
        .clamp(0.0, 255.0) as u8
}

pub fn remove_chroma_key_background(content: &[u8]) -> Result<Vec<u8>> {
    let mut image = image::load_from_memory(content)?.to_rgba8();
    let (width, height) = image.dimensions();
    let mut removable = vec![false; (width * height) as usize];
    let mut queue = VecDeque::new();
    {
        let enqueue =
            |x: u32, y: u32, removable: &mut Vec<bool>, queue: &mut VecDeque<(u32, u32)>| {
                let index = (y * width + x) as usize;
                if !removable[index] && distance(*image.get_pixel(x, y)) <= OUTER_DISTANCE {
                    removable[index] = true;
                    queue.push_back((x, y));
                }
            };
        for x in 0..width {
            enqueue(x, 0, &mut removable, &mut queue);
            if height > 1 {
                enqueue(x, height - 1, &mut removable, &mut queue);
            }
        }
        for y in 1..height.saturating_sub(1) {
            enqueue(0, y, &mut removable, &mut queue);
            if width > 1 {
                enqueue(width - 1, y, &mut removable, &mut queue);
            }
        }
        while let Some((x, y)) = queue.pop_front() {
            if x > 0 {
                enqueue(x - 1, y, &mut removable, &mut queue);
            }
            if x + 1 < width {
                enqueue(x + 1, y, &mut removable, &mut queue);
            }
            if y > 0 {
                enqueue(x, y - 1, &mut removable, &mut queue);
            }
            if y + 1 < height {
                enqueue(x, y + 1, &mut removable, &mut queue);
            }
        }
    }
    for y in 0..height {
        for x in 0..width {
            if !removable[(y * width + x) as usize] {
                continue;
            }
            let pixel = *image.get_pixel(x, y);
            let mut alpha = ((distance(pixel) - INNER_DISTANCE)
                / (OUTER_DISTANCE - INNER_DISTANCE))
                .clamp(0.0, 1.0);
            alpha *= f64::from(pixel[3]) / 255.0;
            let next = if alpha <= 0.03 {
                Rgba([0, 0, 0, 0])
            } else {
                Rgba([
                    recover(pixel[0], KEY[0], alpha),
                    recover(pixel[1], KEY[1], alpha),
                    recover(pixel[2], KEY[2], alpha),
                    (alpha * 255.0).round() as u8,
                ])
            };
            image.put_pixel(x, y, next);
        }
    }
    let mut output = Cursor::new(Vec::new());
    DynamicImage::ImageRgba8(image).write_to(&mut output, ImageFormat::Png)?;
    Ok(output.into_inner())
}

import sys
import tensorflow as tf
import numpy as np

from PIL import Image, ImageDraw, ImageFont
from transformers import AutoTokenizer, TFBertForMaskedLM

# Pre-trained masked language model
MODEL = "bert-base-uncased"

# Number of predictions to generate
K = 3

# Constants for generating attention diagrams
FONT = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 28)
GRID_SIZE = 40
PIXELS_PER_WORD = 200


def main():
    text = input("Text: ")

    # Tokenize input
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    inputs = tokenizer(text, return_tensors="tf")
    mask_token_index = get_mask_token_index(tokenizer.mask_token_id, inputs)
    if mask_token_index is None:
        sys.exit(f"Input must include mask token {tokenizer.mask_token}.")

    # Use model to process input
    model = TFBertForMaskedLM.from_pretrained(MODEL)
    result = model(**inputs, output_attentions=True)

    # Generate predictions
    mask_token_logits = result.logits[0, mask_token_index]
    top_tokens = tf.math.top_k(mask_token_logits, K).indices.numpy()
    for token in top_tokens:
        print(text.replace(tokenizer.mask_token, tokenizer.decode([token])))

    # Obtain the list of tokens from the input IDs
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].numpy())

    # Visualize attentions
    visualize_attentions(tokens, result.attentions)

    
def get_mask_token_index(mask_token_id, inputs):
    
    # The input_ids are located in the inputs["input_ids"] tensor
    input_ids = inputs["input_ids"].numpy()[0]  # Convert to numpy array and select the first item
    # Find the index of the mask token ID
    mask_token_index = np.where(input_ids == mask_token_id)[0]
    return mask_token_index[0] if mask_token_index.size > 0 else None


def get_color_for_attention_score(attention_score):
    # Scale the attention score to a value between 0 and 255
    # Higher scores should result in lighter colors
    gray_value = int(255 * attention_score)
    # Return a tuple with the gray_value repeated three times for R, G, B
    return (gray_value, gray_value, gray_value)


def visualize_attentions(tokens, attentions):
    # Loop over all layers in the attention
    for i, layer in enumerate(attentions):
        # Loop over all heads in the layer
        for j, head in enumerate(layer[0]):  # Beam number is always 0
            # Generate the attention diagram for each head
            generate_diagram(
                layer_number=i+1,  # 1-indexed layer number
                head_number=j+1,  # 1-indexed head number
                tokens=tokens,
                attention_weights=head
            )

def generate_diagram(layer_number, head_number, tokens, attention_weights):
    """
    Generate a diagram representing the self-attention scores for a single
    attention head. The diagram shows one row and column for each of the
    `tokens`, and cells are shaded based on `attention_weights`, with lighter
    cells corresponding to higher attention scores.

    The diagram is saved with a filename that includes both the `layer_number`
    and `head_number`.
    """
    # Create new image
    image_size = GRID_SIZE * len(tokens) + PIXELS_PER_WORD
    img = Image.new("RGBA", (image_size, image_size), "black")
    draw = ImageDraw.Draw(img)

    # Draw each token onto the image
    for i, token in enumerate(tokens):
        # Draw token columns
        token_image = Image.new("RGBA", (image_size, image_size), (0, 0, 0, 0))
        token_draw = ImageDraw.Draw(token_image)
        token_draw.text(
            (image_size - PIXELS_PER_WORD, PIXELS_PER_WORD + i * GRID_SIZE),
            token,
            fill="white",
            font=FONT
        )
        token_image = token_image.rotate(90)
        img.paste(token_image, mask=token_image)

        # Draw token rows
        _, _, width, _ = draw.textbbox((0, 0), token, font=FONT)
        draw.text(
            (PIXELS_PER_WORD - width, PIXELS_PER_WORD + i * GRID_SIZE),
            token,
            fill="white",
            font=FONT
        )

    # Draw each word
    for i in range(len(tokens)):
        y = PIXELS_PER_WORD + i * GRID_SIZE
        for j in range(len(tokens)):
            x = PIXELS_PER_WORD + j * GRID_SIZE
            color = get_color_for_attention_score(attention_weights[i][j])
            draw.rectangle((x, y, x + GRID_SIZE, y + GRID_SIZE), fill=color)

    # Save image
    img.save(f"Attention_Layer{layer_number}_Head{head_number}.png")


if __name__ == "__main__":
    main()

from PIL import Image


def resize_and_pad(image, desired_size=256):
    """
    Resize and pad an image to the desired size.
    
    Parameters:
    image (PIL Image): Image to be resized and padded.
    desired_size (int): Desired size of the output image.
    
    Returns:
    PIL Image: Resized and padded image.
    """
    ratio = float(desired_size) / max(image.size)
    new_size = tuple([int(x * ratio) for x in image.size])
    image = image.resize(new_size, Image.LANCZOS)
    
    # Create a new image and paste the resized on it
    new_image = Image.new("RGB", (desired_size, desired_size))
    new_image.paste(image, ((desired_size - new_size[0]) // 2, (desired_size - new_size[1]) // 2))
    
    return new_image
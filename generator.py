
import argparse
from enum import Enum
import os
from PIL import Image
import cv2
from matplotlib.colors import XKCD_COLORS, hex2color
import numpy as np

ALL_COLORS = {hex2color(val): name.replace('xkcd:', '') for name, val in XKCD_COLORS.items()}
COLORS = [(127,127,0), (0,127,127), (127,0,127)]  # Provide RGB color codes
SPEEDS = [-10,-5,5,10]


class Action(Enum):
    HORIZONTAL = 'horizontal'
    VERTICAL = 'vertical'
    CIRCULAR = 'circular'

def parse_args():
    parser = argparse.ArgumentParser(description='Generate control action dataset with colored shapes as privacy attributes')
    parser.add_argument('--shapedir', type=str, default='data/shapes', help='Path to shapes folder')
    parser.add_argument('--outdir', type=str, default='data/actions', help='Path to generated actions output folder')
    parser.add_argument('--num-videos', type=int, default=10, help='Number of videos to generate')
    parser.add_argument('--fps', type=int, default=16, help='Frames-per-second configuration for video generation')
    parser.add_argument('--frame-width', type=int, default=128, help='Frame width of videos')
    parser.add_argument('--frame-height', type=int, default=128, help='Frame height of videos')
    parser.add_argument('--shape-width', type=int, default=30, help='Width of the shape inside the videos')
    parser.add_argument('--shape-height', type=int, default=30, help='Height of the shape inside the videos')
    args = parser.parse_args()

    assert os.path.isdir(args.shapedir), "Invalid path to shapes folder provided"
    assert len(os.listdir(args.shapedir)) > 0, "Empty shapes directory"

    return args

def closest_color_name(rgb: tuple[int, int, int]):
    """
    Gets closest color name from matplotlibs XKCD_COLORS given a hex RGB tuple

    Args:
        rgb (tuple): the RGB values of the image to whom a name is assigned
    """
    color = hex2color(tuple(t/255 for t in rgb+(255,)))
    distances = np.sum((np.array(list(ALL_COLORS.keys())) - color) ** 2, axis=1)
    closest_idx = np.argmin(distances)
    return ALL_COLORS[list(ALL_COLORS.keys())[closest_idx]]

def get_shape_color_combinations(shapedir: str):
    """
    Creates all possible shape-color combinations

    Args:
        shapedir (str): directory containing shape images
    """
    shapes = os.listdir(shapedir)

    combinations = {}
    for shape in shapes:
        for color in COLORS:
            shape_img = Image.open(os.path.join(shapedir, shape)).convert('RGBA')
            color_layer = Image.new('RGBA', shape_img.size, color)
            colored_shape = Image.composite(color_layer, Image.new('RGBA', shape_img.size, (0, 0, 0, 0)), shape_img)
            combinations[f'{closest_color_name(color)}_{os.path.splitext(shape)[0]}'.replace(' ', '_')] = colored_shape

    return combinations

def get_frames(shape: Image.Image, action: Action, frame_size: tuple[int, int] = (128, 128),
        num_frames: int = 32, speed: int = 1, shape_size: tuple[int, int] = (30, 30), 
        return_initial_pos: bool = False):
    """
    Generates frames for a single action video with the given arguments.

    Args:
        shape (PIL.Image.Image): the shape used for the action
        action (Action): the action to generate
        frame_size (tuple, optional): the size of the frames for the video generation
        num_frame (int, optional): the number of frames to generate
        speed (int, optional): number of pixels per frame for the shape to move
        shape_size (tuple, optional): the size of shape to use in the video
        return_initial_pos (bool, optional): flag to return the randomly sampled initial position alongside frames
    """
    if shape.size != shape_size:
        shape = shape.resize(shape_size)
    shape_w, shape_h = shape.size
    frame_w, frame_h = frame_size

    # Sample random initial position
    initial_pos = (int(np.random.uniform(low=0, high=frame_w - shape_w )), int(np.random.uniform(low=0, high=frame_h - shape_h)))

    positions = []
    if action == Action.HORIZONTAL:
        current_x = initial_pos[0]
        for _ in range(num_frames):
            x = current_x + speed
            if x < 0 or x + shape_w > frame_w:
                speed *= -1
                x = current_x + speed
            positions.append((x, initial_pos[1]))
            current_x = x
    elif action == Action.VERTICAL:
        current_y = initial_pos[1]
        for _ in range(num_frames):
            y = current_y + speed
            if y < 0 or y + shape_h > frame_h:
                speed *= -1
                y = current_y + speed
            positions.append((initial_pos[0], y))
            current_y = y
    elif action == Action.CIRCULAR:
        center_x = (frame_w - shape_w) / 2
        center_y = (frame_h - shape_h) / 2

        radius = np.random.randint(10, min(center_x, center_y)) #  at least 10 radius 
        
        angular_speed = speed / radius
        angle = np.random.uniform(0, 2 * np.pi)  # start randomly on the circle

        for _ in range(num_frames):
            x = center_x + radius * np.cos(angle)
            y = center_y + radius * np.sin(angle)

            positions.append((int(x), int(y)))
            angle += angular_speed

        initial_pos = positions[0]  # different initial_pos was used
    else:
        raise ValueError(f'Unsupported action provided: {action}')
    
    frames = np.zeros((num_frames, frame_h, frame_w, 3), dtype=np.uint8)
    for i, position in enumerate(positions):
        frame = Image.new('RGB', frame_size, (0, 0, 0)) # black bg

        frame.paste(shape, position, shape)
        frames[i] = np.array(frame)
    
    if return_initial_pos:
        return frames, initial_pos
    return frames

def generate_video(outdir: str, shapes_dict: dict, fps: int = 30, **kwargs):
    """
    Generates videos of random shapes doing random actions

    Args:
        outdir (str): directory where to save generated videos
        shapes_dict (dict): dictionary mapping shape names to a PIL Image
        fps (int): frames-per-second setting for generated videos
        kwargs (dict, optional): keyword arguments for get_frames function
    """
    shape_name = np.random.choice(list(shapes_dict.keys()))
    shape = shapes_dict[shape_name]
    action = np.random.choice(list(Action))
    speed = np.random.choice(SPEEDS)  # hardcoded speed options for now

    kwargs['return_initial_pos'] = True
    frames, initial_pos = get_frames(shape=shape, action=action, speed=speed, **kwargs)
    frame_size = frames.shape[1], frames.shape[2]

    counter = 1
    file_extension = '.mp4'
    save_path = os.path.join(outdir, f'{shape_name}_{action.value}_{counter}{file_extension}')
    while os.path.exists(save_path):
        counter += 1
        save_path = os.path.join(outdir, f'{shape_name}_{action.value}_{counter}{file_extension}')

    video = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, frame_size)
    for frame in frames:
        video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    video.release()

    metadata = {
        'save_path': save_path,
        'shape_name': shape_name,
        'action': action.value,
        'speed': speed,
        'initial_pos': initial_pos,
    }
    return metadata


def generate_dataset(outdir: str, shapes_dict: dict, fps: int = 30, n_per_shape_action: int = 15, **kwargs):
    """
    Generates a balanced dataset of shapes doing actions at random speeds and start locations

    Args:
        outdir (str): directory where to save generated videos
        shapes_dict (dict): dictionary mapping shape names to a PIL Image
        fps (int): frames-per-second setting for generated videos
        n_per_class_action (int, optional): number of samples of each shape-action pair to generate
        kwargs (dict, optional): keyword arguments for get_frames function
    """
    for shape_name, shape in shapes_dict.items():
        for action in list(Action):
            os.makedirs(os.path.join(outdir, action.value), exist_ok=True)
            for i in range(n_per_shape_action):
                speed = np.random.choice(SPEEDS)  # hardcoded speed options for now

                frames = get_frames(shape=shape, action=action, speed=speed, **kwargs)
                frame_size = frames.shape[1], frames.shape[2]

                save_path = os.path.join(outdir, action.value, f'{shape_name}_{i+1}.mp4')
                video = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, frame_size)
                for frame in frames:
                    video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                video.release()

if __name__ == "__main__":
    args = parse_args()

    shapes_dict = get_shape_color_combinations(shapedir=args.shapedir)

    # We use the num_videos argument here as the num_per_shape_action input
    generate_dataset(outdir=args.outdir, shapes_dict=shapes_dict, 
                     fps=args.fps, n_per_shape_action=args.num_videos,
                     frame_size=(args.frame_width, args.frame_height), 
                     shape_size=(args.shape_width, args.shape_height))
    print('Generated dataset!')

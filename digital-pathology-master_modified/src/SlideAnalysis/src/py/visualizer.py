###############################################################################
# Creator: Curtis Bechtel
# Date:    4/9/2019
# Project: Senior Design, Digital Pathology
# File:    visualizer.py
# Purpose: This module takes a slide and a set of high powered fields and
#          produces a thumbnail image of the full slide and and one image of
#          each high powered field.
# Associated Files: ......
###############################################################################

import json
from os import path
from PIL import Image
from openslide import OpenSlide





def get_opt(opts, keyword, default):
    if keyword in opts:
        return opts[keyword]
    else:
        return default

def get_thumbnail(slide, scale, **opts):
    """
    TODO
    """
    
    level = get_opt(opts, 'level', 0)
    
    width, height = slide.dimensions
    new_width  = int((width  + scale - 1) // scale)
    new_height = int((height + scale - 1) // scale)
    thumbnail = Image.new('RGB', (new_width, new_height))
    
    for y in range(new_height):
        for x in range(new_width):
            location = (x * scale, y * scale)
            level = 0
            size = (1, 1)
            
            region = slide.read_region(location, level, size)
            thumbnail.paste(region, (x, y))
    
    return thumbnail

def visualize(slide, hpfs, cells, **opts):
    """
    TODO
    """
    print(json.dumps('started visualizing'))
    directory   = get_opt(opts, 'dir', '.')
    basename    = get_opt(opts, 'basename', 'slide')
    level       = get_opt(opts, 'level', 0)
    hpf_radius  = get_opt(opts, 'hpf_radius', 5500)
    cell_radius = get_opt(opts, 'cell_radius', 25)
    max_width   = get_opt(opts, 'max_width', 1100)
    max_height  = get_opt(opts, 'max_height', 1100)
    
    # Get the scale factor of the thumbnail slide
    width, height = slide.level_dimensions[level]
    if width > max_width or height > max_height:
        thumbnail_scale = max(width // max_width, height // max_height)
    
    # Create and save the thumbnail slide
    full_img = get_thumbnail(slide, thumbnail_scale, level=level)
    full_path = path.join(directory, basename + '_full.png')
    full_img.save(full_path)
    
    # Create the JSON object storing data about this slide
    data = {
        'slide_path': full_path,
        'hpf_radius': hpf_radius // thumbnail_scale,
        'cell_radius': cell_radius,
        'hpfs': [],
    }
    
    # Group cells by which HPF they belong to
    hpf_cells = [[] for hpf in hpfs]
    for (cell_x, cell_y) in cells:
        for i, (hpf_x, hpf_y) in enumerate(hpfs):
            if (cell_x - hpf_x) ** 2 + (cell_y - hpf_y) ** 2 <= hpf_radius ** 2:
                hpf_cells[i].append((cell_x, cell_y))
    
    # For each HPF and the cells in that HPF, create and save an image of it
    # and add information about it to the JSON object
    for i, ((hpf_x, hpf_y), cells) in enumerate(zip(hpfs, hpf_cells)):
        corner_x, corner_y = (hpf_x - hpf_radius, hpf_y - hpf_radius)
        size = (2 * hpf_radius, 2 * hpf_radius)
        
        hpf_img = slide.read_region((corner_x, corner_y), level, size)
        hpf_path = path.join(directory, basename + '_hpf_{}.png'.format(i + 1))
        print(json.dumps({'created file': hpf_path}))
        hpf_img.save(hpf_path)
        
        data['hpfs'].append({
            'path': hpf_path,
            'position': (hpf_x // thumbnail_scale, hpf_y // thumbnail_scale),
            'cells': [(x - corner_x, y - corner_y) for x, y in cells],
        })
    
    return data




def main(args):
    hpfs = [
        (10000, 10000),
        (20000, 20000),
        (30000, 30000),
        (40000, 40000),
    ]
    
    cells = [
        (20000 +   0, 20000 +   0),
        (20000 + 500, 20000 + 500),
        (20000 - 500, 20000 - 500),
        
        (30000 +   0, 30000 +   0),
        (30000 +   0, 30000 + 500),
        (30000 + 500, 30000 +   0),
        (30000 +   0, 30000 - 500),
        (30000 - 500, 30000 +   0),
        
        (40000 +   0, 40000 +   0),
        (40000 +   0, 40000 + 500),
        (40000 + 500, 40000 + 500),
        (40000 + 500, 40000 +   0),
        (40000 +   0, 40000 - 500),
        (40000 - 500, 40000 - 500),
        (40000 - 500, 40000 +   0),
    ]
    
    import time
    t = 0
    def clock_start():
        nonlocal t
        t = time.clock()
    def clock_lap():
        nonlocal t
        print('lap', time.clock() - t)
        t = time.clock()
    
    opts = {
        'hpf_radius': 1000
    }
    
    for path in args:
        slide = OpenSlide(path)
        clock_start()
        
        print(visualize(slide, hpfs, cells, max_width= 128, basename='slide_0128', **opts))
        clock_lap()
        
        print(visualize(slide, hpfs, cells, max_width= 256, basename='slide_0256', **opts))
        clock_lap()
        
        print(visualize(slide, hpfs, cells, max_width= 512, basename='slide_0512', **opts))
        clock_lap()
        
        print(visualize(slide, hpfs, cells, max_width=1024, basename='slide_1024', **opts))
        clock_lap()
        
        print(visualize(slide, hpfs, cells, max_width=2048, basename='slide_2048', **opts))
        clock_lap()

if __name__ == '__main__':
    import sys
    exit(main(sys.argv[1:]))


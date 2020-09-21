import os
import sys
import argparse

from DigiPathUtils import decompose_file, save_img


def parseArgs():
    ''' 
    Parse command line arguments:
    server log [port]
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('file',                                       help='Path to .svs file')
    parser.add_argument('--delta',  '-d',   default=256,    type=int, help='Sub-sample size in pixels')
    parser.add_argument('--col',    '-c',   default=0,      type=int, help='Starting column coordinate (top left pixel)')
    parser.add_argument('--row',    '-r',   default=0,      type=int, help='Starting row coordinate (top left pixel)')
    parser.add_argument('--across', '-x',   default=None,   type=int, help='Number of sub-samples across')
    parser.add_argument('--down',   '-y',   default=None,   type=int, help='Number of sub-samples down')
    parser.add_argument('--outdir', '-o',   default='./',   type=str, help='Output directory for image files')
    parser.add_argument('--resize', '-s',   default=None,   type=int, help='Resize images')
    parser.add_argument('--png',            default=False,  action='store_true', help='Use .png instead of .jpg')
    args = parser.parse_args()
    return args


def main():
    args = parseArgs()
    decompose_file(
        filename=args.file, 
        delta=args.delta, 
        begin=[args.col, args.row], 
        n=[args.across, args.down],
        out_dir=args.outdir,
        size=None if args.resize is None else [args.resize, args.resize],
        JPEG=not args.png
    )

if __name__ == "__main__":
    main()
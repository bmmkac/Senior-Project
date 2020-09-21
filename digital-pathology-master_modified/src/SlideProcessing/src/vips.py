import os
import subprocess
import argparse
from DigiPathDb import DigiPathDbWrapper

def parseArgs():
    ''' 
    Parse command line arguments
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('file', help='Path to .svs file')
    parser.add_argument('--outdir', '-o',   default='/home/images/vips/jpg100qcompression/',   type=str, help='Output directory for image files')
    args = parser.parse_args()
    return args

def main():
    args = parseArgs()
    
    slide_name = os.path.splitext(os.path.basename(args.file))[0]
    out_dir = os.path.join(args.outdir, slide_name)

    path = os.path.abspath(args.file)
    with DigiPathDbWrapper() as db:
        db.new_slide(slide_name, path)

    cmd = ['vips', 'dzsave', path, out_dir, '--background', '0', '--centre', '--layout', 'google']
    result = subprocess.run(cmd, stdout=subprocess.PIPE)
    print(result.stdout.decode('utf-8'))

if __name__ == "__main__":
    main()

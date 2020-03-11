#!/bin/bash

python color_calib.py --method projective --headless --filename color_checker.jpg --prefix proj_std
python color_calib.py --method poly --headless --poly_degree 3 --max_nfev 256 --filename color_checker.jpg --prefix poly_fast
python color_calib.py --method poly --headless --poly_degree 3 --max_nfev 256 --filename 20200220_120820.jpg --prefix poly_blue_fast
python color_calib.py --method poly --headless --filename color_checker.jpg --prefix poly_std
python color_calib.py --method projective --headless --filename 20200213_103542.jpg --prefix proj_noise
python color_calib.py --method projective --headless --filename 20200220_120820.jpg --prefix proj_blue
python color_calib.py --method projective --headless --filename 20200220_120829.jpg --prefix proj_yellow
python color_calib.py --method poly --poly_degree 6 --headless --filename 20200220_120820.jpg --prefix poly_blue
python color_calib.py --method poly --poly_degree 6 --headless --filename 20200220_120829.jpg --prefix poly_yellow
python color_calib.py --method poly --poly_degree 6 --headless --filename 20200303_114812.jpg --prefix poly_skew


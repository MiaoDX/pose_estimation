#!/usr/bin/python
#! -*- encoding: utf-8 -*-

# This file is changed from OpenMVG (Open Multiple View Geometry) C++ library example.


import os
import subprocess
import sys


# Indicate the openMVG binary directory
# OPENMVG_SFM_BIN = "H:/projects/SLAM/sfm/openMVG_example/openMVG_SfM_official/build/SfM/Release"
# OPENMVG_Localization_BIN = "H:/projects/SLAM/sfm/openMVG_example/openMVG_SfM_official/build/Localization/Release"

OPENMVG_SFM_BIN = "H:/projects/SLAM/sfm/openMVG_develop/src/release/Windows-AMD64-Release/Release"
OPENMVG_Localization_BIN = OPENMVG_SFM_BIN

WORK_DIR = "H:/projects/SLAM/pose_estimation/SfM_Aided/"
OPENMVG_SFM_MINE_BIN = WORK_DIR+"build/SfM/Release"
OPENMVG_Geodesy_MINE_BIN = WORK_DIR+"build/Geodesy/Release"


def cvt_SfM_data(input_dir, output_dir, filename):
  basename = os.path.basename(filename)
  f_no_ext, ext_f = os.path.splitext(basename)
  input_file = os.path.join(input_dir, filename)
  
  for ext in ['bin', 'json', 'ply']:
    if ext == ext_f:
      continue
    output_file = os.path.join(output_dir, f_no_ext+'.'+ext)
    pCvt = subprocess.Popen( [os.path.join(OPENMVG_SFM_BIN, "openMVG_main_ConvertSfM_DataFormat"),  "-i", input_file, "-o", output_file] )
    pCvt.wait()


def incremental_SfM_pipeline(dataset_dir, output_dir, K_value):
  init_dir = os.path.join(dataset_dir, "sfm_init_data") # folder of train images, should make sure
  faked_gps_path = os.path.join(dataset_dir, "fake_gps_file.txt")

  if not os.path.exists(dataset_dir) or not os.path.exists(init_dir):
    print("Seems no valid dataset")
    exit()

  if not os.path.exists(faked_gps_path):
    print("no faked gps provided")
    exit()


  matches_dir = os.path.join(output_dir, "matches")
  reconstruction_dir = os.path.join(output_dir, "reconstruction_sequential")

  print ("Using dataset dir  : ", dataset_dir)
  print ("      images in {} for initialization".format(init_dir))
  print ("      output_dir : ", output_dir)
  print ("      matches_dir : ", matches_dir)
  print ("      reconstruction_dir : ", reconstruction_dir) 

  # Create the ouput/matches folder if not present
  if not os.path.exists(output_dir):
    os.mkdir(output_dir)
  if not os.path.exists(matches_dir):
    os.mkdir(matches_dir)

  print ("1. Intrinsics analysis")
  pIntrisics = subprocess.Popen( [os.path.join(OPENMVG_SFM_MINE_BIN, "SfMInit_ImageListing"),  "--imageDirectory", init_dir, "--outputDirectory", matches_dir, "--intrinsics", K_value] )
  pIntrisics.wait()
  
  print("1.x Geodesy, add position prior")
  pGeodesy = subprocess.Popen( [os.path.join(OPENMVG_Geodesy_MINE_BIN, "registration_faked_gps_position"),  "--input_file", matches_dir+"/sfm_data.json", "--output_file", matches_dir+"/sfm_data.json", "--faked_gps_path", faked_gps_path] )
  pGeodesy.wait()
  

  # time, 33s for 10 images, an improvement to be needed
  print ("2. Compute features")
  pFeatures = subprocess.Popen( [os.path.join(OPENMVG_SFM_BIN, "openMVG_main_ComputeFeatures"),  "--input_file", matches_dir+"/sfm_data.json", "--outdir", matches_dir, "--describerMethod", "SIFT", "--numThreads", "7"] ) # threads num is related to CPU
  pFeatures.wait()
  
  # with guidedmatching, 32s will be used, really long
  print ("3. Compute matches")
  pMatches = subprocess.Popen( [os.path.join(OPENMVG_SFM_BIN, "openMVG_main_ComputeMatches"),  "--input_file", matches_dir+"/sfm_data.json", "--out_dir", matches_dir] )
  # pMatches = subprocess.Popen( [os.path.join(OPENMVG_SFM_BIN, "openMVG_main_ComputeMatches"),  "--input_file", matches_dir+"/sfm_data.json", "--out_dir", matches_dir, "--geometric_model", "e", "--guided_matching", "1"] )
  pMatches.wait()
  
  

  # Create the reconstruction if not present
  if not os.path.exists(reconstruction_dir):
      os.mkdir(reconstruction_dir)

  print ("4. Do Sequential/Incremental reconstruction")
  pRecons = subprocess.Popen( [os.path.join(OPENMVG_SFM_BIN, "openMVG_main_IncrementalSfM"),  "--input_file", matches_dir+"/sfm_data.json", "--matchdir", matches_dir, "--outdir", reconstruction_dir, "--refineIntrinsics", "NONE", "--prior_usage"] )
  pRecons.wait()
  

  print("4.1 Convert format for easy looking")
  cvt_SfM_data(reconstruction_dir, reconstruction_dir, "sfm_data.bin")
  
  # print ("4.2 Colorize Structure") # no need to do this, to save time
  # pRecons = subprocess.Popen( [os.path.join(OPENMVG_SFM_BIN, "openMVG_main_ComputeSfM_DataColor"),  "-i", reconstruction_dir+"/"+incremental_SfM_data_name, "-o", os.path.join(reconstruction_dir,"colorized_incremental_SfM.ply")] )
  # pRecons.wait()

  # optional, compute final valid structure from the known camera poses, a refine part
  # Note the matches.e.bin or matches.f.bin
  print ("5. Structure from Known Poses (robust triangulation)")
  pRecons = subprocess.Popen( [os.path.join(OPENMVG_SFM_BIN, "openMVG_main_ComputeStructureFromKnownPoses"),  "--input_file", reconstruction_dir+"/sfm_data.bin", "--match_dir", matches_dir, "--output_file", os.path.join(reconstruction_dir,"robust.bin")] )
  pRecons.wait()
  

  print("5.1 Convert format for easy looking")
  cvt_SfM_data(reconstruction_dir, reconstruction_dir, "robust.bin")

  # print ("5.2 Colorize Structure")
  # pRecons = subprocess.Popen( [os.path.join(OPENMVG_SFM_BIN, "openMVG_main_ComputeSfM_DataColor"),  "-i", reconstruction_dir+"/robust.bin", "-o", os.path.join(reconstruction_dir,"robust_colorized.ply")] )
  # pRecons.wait()



def localization_pipeline(dataset_dir, output_dir, 
  incremental_SfM_data_name,
  reference_im_name="reference_2.jpg",
  query_im_name="query_1.jpg"
  ):

  query_dir = os.path.join(dataset_dir, "sfm_query_data")

  matches_dir = os.path.join(output_dir, "matches")
  reconstruction_dir = os.path.join(output_dir, "reconstruction_sequential")
  localization_dir = os.path.join(output_dir, "Localization")

  log_file = os.path.join(localization_dir,'localization_log_'+reference_im_name+'_'+query_im_name+'.txt')


  query_im_path = os.path.join(query_dir, query_im_name)

  print ("Using dataset dir  : ", dataset_dir)
  print ("      output_dir : ", output_dir)
  print ("      localization_dir : ", localization_dir)


  # Create the ouput/matches folder if not present
  if not os.path.exists(output_dir) or not os.path.exists(matches_dir):
    print("seems we have not do initial SfM")
    exit()
  if not os.path.exists(localization_dir):
        os.mkdir(localization_dir)

  if os.path.exists(log_file):
        os.remove(log_file)

  ##NOTE: the `-s –single_intrinsics` is rather important, it will not try to BA the intrinsics, leave our methods more stable

  print ("1. Localization ..")
  pLocal = subprocess.Popen( [os.path.join(OPENMVG_Localization_BIN, "openMVG_main_SfM_Localization"),  "--input_file",  reconstruction_dir+"/"+incremental_SfM_data_name, "--match_dir", matches_dir, "--out_dir", localization_dir, "--match_out_dir", localization_dir, "--query_image_dir", query_im_path, "--single_intrinsics"], shell=True,stdout = open(log_file,'w') )
  pLocal.wait()

  print ("2. Calc relative pose of {}, regrding to reference image {}".format(query_im_name, reference_im_name))
  pRelativePose = subprocess.Popen( [os.path.join(OPENMVG_SFM_MINE_BIN, "relativePosePair_test"),  "--input_file",  localization_dir+"/sfm_data_expanded.json", "--reference_im_name", reference_im_name, "--query_im_name", query_im_name], shell=True,stdout = open(log_file,'a+') )
  pRelativePose.wait()


if __name__ == "__main__":
  """
  Some user defined variables
  """
  # dataset_dir = WORK_DIR+"dataset_cartoon_1/"
  # output_dir = WORK_DIR + "sequential_cartoon_1_jpg"
  # # dataset_dir = WORK_DIR+"dataset_Marx_1/"
  # # output_dir = WORK_DIR + "sequential_Marx_1_jpg"
  # K_value = "8607.8639;0;2880.72115;0;8605.4303;1913.87935;0;0;1" # K for big images, jpg
  # suffix = ".jpg"

  dataset_dir = WORK_DIR+"dataset_cartoon_1_bmp/"
  output_dir = WORK_DIR + "sequential_cartoon_1_bmp"
  K_value = "1444.29449;0.0;482.68264;0.0;1444.79783;319.3993;0.0;0.0;1" # K for small images, bmp
  suffix = ".jpg" # we should convert the suffix first


  # incremental_SfM_pipeline(dataset_dir, output_dir, K_value)
  

  # localization_pipeline(dataset_dir, output_dir, 
  # "robust.bin",
  # reference_im_name="reference_2.jpg",
  # query_im_name="query_1.jpg"
  # )


  strs = [str(x) for x in range(1, 10)]
  # strs.extend(['1a', '1b', '1c', '1d', '4a', '4b', '7a', '7b'])
  strs = sorted(strs)

  for im in strs:
    query = "query_"+im+suffix
    localization_pipeline(dataset_dir, output_dir, 
    "robust.bin",
    reference_im_name="reference_1"+suffix,
    query_im_name=query
    )

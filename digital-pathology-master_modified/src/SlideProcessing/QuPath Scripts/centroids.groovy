import qupath.lib.scripting.QPEx // for printing loaded project image file names

// Setup output file name
def outputDir = "/home/qpproj/"
def outputFileName = outputDir + "centroid_positions.txt"
print "WARNING - Remember to delete the output file if it already exists, as this just appends!"
File f = new File(outputFileName)

// Part 1 - Detect all objects
// Select entire region
clearDetections()
createSelectAllObject(true)

// Run cell analysis. These lines were obtained using the "Create Script" button in the Workflow menu
// after running cell detection with default settings
setImageType('BRIGHTFIELD_H_DAB');
setColorDeconvolutionStains('{"Name" : "H-DAB default", "Stain 1" : "Hematoxylin", "Values 1" : "0.65111 0.70119 0.29049 ", "Stain 2" : "DAB", "Values 2" : "0.26917 0.56824 0.77759 ", "Background" : " 255 255 255 "}');
selectAnnotations();
runPlugin('qupath.imagej.detect.nuclei.WatershedCellDetection', '{"detectionImageBrightfield": "Hematoxylin OD",  "backgroundRadius": 15.0,  "medianRadius": 0.0,  "sigma": 3.0,  "minArea": 10.0,  "maxArea": 1000.0,  "threshold": 0.2,  "maxBackground": 2.0,  "watershedPostProcess": true,  "excludeDAB": false,  "cellExpansion": 5.0,  "includeNuclei": true,  "smoothBoundaries": true,  "makeMeasurements": true}');


// Part 2 - Export the centroids for all detections, along with their classifications

// Set this to true to use a nucleus ROI, if available
boolean useNucleusROI = true

// Start building a String with a header
sb = new StringBuilder("")

// Loop through detections
int n = 0
for (detection in getDetectionObjects()) {
    def roi = detection.getROI()
    // Use a Groovy metaClass trick to check if we can get a nucleus ROI... if we need to
    // (could also use Java's instanceof qupath.lib.objects.PathCellObject)
    if (useNucleusROI && detection.metaClass.respondsTo(detection, "getNucleusROI") && detection.getNucleusROI() != null)
        roi = detection.getNucleusROI()
    // ROI shouldn't be null... but still feel I should check...
    if (roi == null)
        continue
    // Get class
    def pathClass = detection.getPathClass()
    def className = pathClass == null ? "" : pathClass.getName()
    // Get centroid
    double cx = roi.getCentroidX()
    double cy = roi.getCentroidY()
    
    // Append to String
    sb.append(String.format("%.2f\t%.2f\n", cx, cy))
    // Count
    n++
}

// Get current image name
def imageData = QPEx.getCurrentImageData()
def server = imageData.getServer()
String imageName = server.getPath()

// Print filename and centroids
f.append(imageName + "\n") 
f.append(sb)
println("Output saved to " + f)
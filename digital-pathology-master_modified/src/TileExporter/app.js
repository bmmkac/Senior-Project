
//==============================================================================
// Modules
//==============================================================================

/*
Local javascript module
*/
// const idk = require("./path...");

/*
The path module for manipulating file paths
*/
const path = require("path");

/*
The file system module for reading/writing files
*/
const fs = require("fs");

/*
The Express module for asynchrnonous routing
*/
const express = require("express");

/*
The BodyParser module for parsing POST request bodies into JSON
*/
const body_parser = require("body-parser");





//==============================================================================
// Server Constants
//==============================================================================

/*
The port number for this server to use. It is set to the value of the
environment variable PORT, or 8080 if PORT isn't set.
*/
const PORT = process.env.PORT || 8000;

/*
The directory for files with public access such as static HTML and CSS
files. These will be served statically from the server to the client.
*/
const PUBLIC_DIR = path.join(__dirname, "public");

/*
The directory for storing exported tile data.
*/
const QPPROJ_DIR = '/home/qpproj';

/*
The directory where tiled images are stored
*/
const IMAGE_DIR = '/home/images/vips/jpg100qcompression';


//==============================================================================
// App Object
//==============================================================================

/*
Initialize an instance of the server with BodyParser and static serving of any
files in the public directory
*/
const app = express();

app.use(express.static(PUBLIC_DIR));
app.use(body_parser.json());
app.use(body_parser.urlencoded({
    "extended": true
}));




//==============================================================================
// Utility Functions
//==============================================================================

function isString(obj) {
    return typeof obj == "string";
}




//==============================================================================
// HTTP Routes
//==============================================================================

/*
Creates and returns a callback function for automatically sending a HTTP
response containing an error code and an object
*/
function createErrorCheckCallback(res) {
    return function (err, result) {
        if (isString(err)) {
            res.status(400);
            res.send(err);

        } else if (err) {
            res.status(500);
            res.send("Unknown server error.");

        } else if (typeof result === "undefined" || result === null) {
            res.status(200);
            res.send("Success.");

        } else {
            res.status(200);
            res.send(result);
        }
    };
}

function arrayUnion(arr1, arr2) {
    return Array.from(new Set(arr1.concat(arr2)));
}

function fromQPProjFormat(data) {
    var arr = [];
    for (var i in data.images) {
        arr.push(data.images[i].path);
    }
    return arr;
}

function toQPProjFormat(arr) {
    var t = Date.now();

    var data = {};
    data['createTimestamp'] = t;
    data['modifyTimestamp'] = t;

    var images = [];
    for (var i in arr) {
        var img = {};
        img['path'] = arr[i];
        img['name'] = arr[i].split('/').pop();
        images.push(img);
    }

    data['images'] = images;
    return data;
}

function updateFileArray(filename, arr, callback) {
    fs.access(filename, fs.F_OK | fs.R_OK | fs.W_OK, function (err) {
        if (err) {
            // The file doesn't exist, write a new one
            var data = toQPProjFormat(arr);
            fs.writeFile(filename, JSON.stringify(data), function (err) {
                if (err) {
                    return callback("Failed to create new file '" + filename + "'.");
                }
                fs.chmod(filename, 0666);
                callback(false);
            });

        } else {
            // The file already exists. Update the array it contains
            fs.readFile(filename, function (err, data) {
                if (err) {
                    return callback("Failed to read from file '" + filename + "'.");
                }

                data = JSON.parse(data);
                arr = arrayUnion(arr, fromQPProjFormat(data));
                data = toQPProjFormat(arr);
                fs.writeFile(filename, JSON.stringify(data), function (err) {
                    if (err) {
                        return callback("Failed to write to file '" + filename + "'");
                    }

                    callback(false);
                });
            });
        }
    });
}

/*
Route for the client to get the maximum zoom level for a given slide
*/
app.get('/max-zoom', function (req, res) {
    var callback = createErrorCheckCallback(res);

    console.log("GET /max-zoom " + req.query.slide);

    var dir = path.join(IMAGE_DIR, req.query.slide);

    fs.readdir(dir, function (err, items) {
        if (err) {
            return callback("Failed to open directory");
        }

        var max = 0;

        for (var i = 0; i < items.length; i++) {
            var zoom = Number(items[i]);
            if (!isNaN(zoom) && zoom > max)
                max = zoom;
        }

        callback(false, max.toString());
    });
});

/*
Route for the client to get a list of all available slides to view
*/
app.get('/all-slides', function (req, res) {
    var callback = createErrorCheckCallback(res);

    console.log("GET /all-slides");

    fs.readdir(IMAGE_DIR, function (err, items) {
        if (err) {
            return callback("Failed to open directory");
        }

        var slides = {};
        slides['slides'] = [];
        for (var i = 0; i < items.length; i++) {
            slides['slides'].push(items[i]);
        }

        callback(false, slides);
    });
});

/*
Route for the client to export a list of tiles
*/
app.post("/export-tiles", function (req, res) {
    var callback = createErrorCheckCallback(res);
    var body = req.body;

    console.log("POST /export-tiles " + JSON.stringify(body));

    if (!isString(body.slide)) {
        return callback("Invalid POST request: 'slide' must be a string.");
    }

    if (!Array.isArray(body.images)) {
        return callback("Invalid POST request: 'images' must be an array.");
    }

    for (var i = 0; i < body.images.length; i++) {
        if (!isString(body.images[i])) {
            return callback("Invalid POST request: 'images' must be an array of strings.");
        }
    }

    var dir = path.join(QPPROJ_DIR, body.slide); // put qpproj file into a subdirectory
    if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir);
        fs.chmod(dir, 0777);
    }
    var filename = path.join(dir, body.slide + ".qpproj");

    updateFileArray(filename, body.images, callback);
});



//==============================================================================
// Finally
//==============================================================================

/*
Start the server
*/
app.listen(PORT, function () {
    console.log("Started server on port " + PORT + ".");
});


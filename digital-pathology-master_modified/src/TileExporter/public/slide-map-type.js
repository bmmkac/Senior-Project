var TILE_SIZE = 256;
var DU_SERVER_ROOT = 'http://129.25.13.204';
var ROOT_DIR = DU_SERVER_ROOT + '/images/vips/jpg100qcompression';
var TILE_EXPORTER_ENDPOINT = DU_SERVER_ROOT + ':8000/export-tiles';
var MAX_ZOOM_ENDPOINT = DU_SERVER_ROOT + ':8000/max-zoom';
var ALL_SLIDES_ENDPOINT = DU_SERVER_ROOT + ':8000/all-slides';

var selectedTiles = {};
var maxZoom = -1;
var slideName = '';

// ************************* //
//            Menu           //
// ************************* //

var menu = document.getElementById('menu');

// Add slide name as title
var title = document.createElement('h1');
title.innerText = getSlideName();
title.style.textIndent = "10px";
menu.appendChild(title);

// create bulleted list of links to other slides
var all_slides = getAllSlides();

var list = document.createElement('ul');
for(var i = 0; i < all_slides.length; i++) 
{
    var slide = all_slides[i];
    var url = DU_SERVER_ROOT + ':8000?slide=' + slide;
    var item = document.createElement('li');
    
    var a = document.createElement('a');
    var text = document.createTextNode(slide);
    a.appendChild(text);
    a.title = slide;
    a.href = url;
    
    item.appendChild(a);
    list.appendChild(item);
}

menu.appendChild(list);


// ************************* //
//            Map            //
// ************************* //
var map = new google.maps.Map(document.getElementById('map'), {
    center: { lat: 0, lng: 0 },
    zoom: 0,
    streetViewControl: false,
    mapTypeControl: false
});

// Left click Listener to select a single tile
map.addListener('click', function (e) {
    // get max zoom level
    var zoom = getMaxZoom();
    var scale = 1 << zoom;

    // get world coordinate from given latitude and longitude
    var worldCoordinate = map.getProjection().fromLatLngToPoint(e.latLng);

    // get tile coordinate at max zoom level
    var tileCoordinate = new google.maps.Point(
        Math.floor(worldCoordinate.x * scale / TILE_SIZE),
        Math.floor(worldCoordinate.y * scale / TILE_SIZE));

    selectTile(tileCoordinate, map, zoom);
});

// Right click listener to select a group of tiles
map.addListener('rightclick', function (e) {
    // get current zoom level
    var zoom = map.getZoom();
    var scale = 1 << zoom;

    // get world coordinate from given latitude and longitude
    var worldCoordinate = map.getProjection().fromLatLngToPoint(e.latLng);

    // get tile coordinate at current zoom level
    var tileCoordinate = new google.maps.Point(
        Math.floor(worldCoordinate.x * scale / TILE_SIZE),
        Math.floor(worldCoordinate.y * scale / TILE_SIZE));

    selectMultiTile(tileCoordinate, map, zoom, getMaxZoom());
});


// ************************* //
//   Slide Image Map Type    //
// ************************* //
var slideMapType = new google.maps.ImageMapType({
    getTileUrl: function (coord, zoom) {
        var tileRange = 1 << zoom;
        if (coord.y < 0 || coord.y >= tileRange || coord.x < 0 || coord.x >= tileRange)
            return null;

        return getTileImage(zoom, coord);
    },
    tileSize: new google.maps.Size(TILE_SIZE, TILE_SIZE),
    maxZoom: getMaxZoom(),
    minZoom: 2,
    name: 'Slide'
});

map.mapTypes.set('slide', slideMapType);
map.setMapTypeId('slide');


// ************************* //
//       Grid Overlay        //
// ************************* //
function GridOverlayType(tileSize) {
    this.tileSize = tileSize;
}

GridOverlayType.prototype.getTile = function (coord, zoom, ownerDocument) {
    var div = ownerDocument.createElement('div');
    div.innerHTML = coord;
    div.style.width = this.tileSize.width + 'px';
    div.style.height = this.tileSize.height + 'px';
    div.style.fontSize = '10';
    div.style.borderStyle = 'solid';
    div.style.borderWidth = '1px';
    div.style.borderColor = '#AAAAAA';
    return div;
};

var gridOverlay = new GridOverlayType(new google.maps.Size(TILE_SIZE, TILE_SIZE));

// ************************* //
//      Custom Controls      //
// ************************* //
function CustomControl(controlDiv, map) {

    // ************************* //
    //    Toggle Grid Control    //
    // ************************* //

    var toggleGridUI = document.createElement('div');
    toggleGridUI.id = 'toggleGridUI';
    toggleGridUI.title = 'Click here to toggle grid';
    controlDiv.appendChild(toggleGridUI);

    var toggleGridText = document.createElement('div');
    toggleGridText.id = 'toggleGridText';
    toggleGridText.innerHTML = 'Toggle grid';
    toggleGridUI.appendChild(toggleGridText);

    toggleGridUI.addEventListener('click', function () {
        if (map.overlayMapTypes.length == 0) {
            map.overlayMapTypes.push(gridOverlay);
            toggleGridUI.style.borderColor = '#f66';
        }
        else {
            map.overlayMapTypes.clear();
            toggleGridUI.style.borderColor = '#fff';
        }
    });

    // ************************* //
    // Clear Selections Control  //
    // ************************* //

    var clearSelectionUI = document.createElement('div');
    clearSelectionUI.id = 'clearSelectionUI';
    clearSelectionUI.title = 'Click to clear selected tiles';
    controlDiv.appendChild(clearSelectionUI);

    var clearSelectionText = document.createElement('div');
    clearSelectionText.id = 'clearSelectionText';
    clearSelectionText.innerHTML = 'Clear selections';
    clearSelectionUI.appendChild(clearSelectionText);

    clearSelectionUI.addEventListener('click', function () {
        for (var tile in selectedTiles) {
            // remove the rectangle from the map and delete it
            var rect = selectedTiles[tile].rect;
            rect.setMap(null);
            delete selectedTiles[tile];
        }
    });

    // ************************* //
    // Export Selections Control //
    // ************************* //

    var exportSelectionUI = document.createElement('div');
    exportSelectionUI.id = 'exportSelectionUI';
    exportSelectionUI.title = 'Click to export selected tiles';
    controlDiv.appendChild(exportSelectionUI);

    var exportSelectionText = document.createElement('div');
    exportSelectionText.id = 'exportSelectionText';
    exportSelectionText.innerHTML = 'Export selections';
    exportSelectionUI.appendChild(exportSelectionText);

    exportSelectionUI.addEventListener('click', function () {
        var files = [];
        for (var tile in selectedTiles) {
            var file = getTileImage(getMaxZoom(), selectedTiles[tile].coord);
            file = file.replace(DU_SERVER_ROOT, '/home');
            files.push(file);
        }
        if(files.length == 0)
            return;

        var data = {};
        data["slide"] = getSlideName();
        data["images"] = files;

        console.log(JSON.stringify(data));

        var xhr = new XMLHttpRequest();
        xhr.open("POST", TILE_EXPORTER_ENDPOINT, true);
        xhr.setRequestHeader("Content-Type", "application/json");
        xhr.send(JSON.stringify(data));
    });
}

var controlDiv = document.createElement('div');
var customControl = new CustomControl(controlDiv, map);
map.controls[google.maps.ControlPosition.BOTTOM_LEFT].push(controlDiv);


// ************************* //
//        Utilities          //
// ************************* //

// Handle tile selections
function selectTile(tileCoordinate, map, zoom) {
    var scale = 1 << zoom;

    // check if tile is already selected
    if (tileCoordinate in selectedTiles) {
        // remove the rectangle from the map and delete it
        var rect = selectedTiles[tileCoordinate].rect;
        rect.setMap(null);
        delete selectedTiles[tileCoordinate];
    }
    else {
        // add it to the selected tiles
        selectedTiles[tileCoordinate] = new Object();
        selectedTiles[tileCoordinate].rect = createTileRectangle(tileCoordinate, map, scale);
        selectedTiles[tileCoordinate].coord = tileCoordinate;
    }
}

// Handle multi-tile selections
function selectMultiTile(tileCoordinate, map, zoom, maxZoom) {
    if (zoom == maxZoom) {
        selectTile(tileCoordinate, map, zoom);
        return;
    }

    var scale = 1 << zoom;

    // get world coordinates for the top left corner of the tile
    var topLeftWorldCoord = new google.maps.Point(
        tileCoordinate.x * TILE_SIZE / scale,
        tileCoordinate.y * TILE_SIZE / scale);

    // get top left tile coordinate at next zoom level
    scale <<= 1;
    var zoom_x = Math.floor(topLeftWorldCoord.x * scale / TILE_SIZE);
    var zoom_y = Math.floor(topLeftWorldCoord.y * scale / TILE_SIZE);
    
    // recurse on the four tiles that sub-divide the current tile
    selectMultiTile(new google.maps.Point(zoom_x, zoom_y), map, zoom + 1, maxZoom);
    selectMultiTile(new google.maps.Point(zoom_x + 1, zoom_y), map, zoom + 1, maxZoom);
    selectMultiTile(new google.maps.Point(zoom_x, zoom_y + 1), map, zoom + 1, maxZoom);
    selectMultiTile(new google.maps.Point(zoom_x + 1, zoom_y + 1), map, zoom + 1, maxZoom);
}

// Create a rectangle on the map for the selected tile
function createTileRectangle(tileCoordinate, map, scale) {
    // get world coordinates for the top left and bottom right corners of the tile
    var topLeftWorldCoord = new google.maps.Point(
        tileCoordinate.x * TILE_SIZE / scale,
        tileCoordinate.y * TILE_SIZE / scale);
    var bottomRightWorldCoord = new google.maps.Point(
        ((tileCoordinate.x + 1) * TILE_SIZE - 1) / scale,
        ((tileCoordinate.y + 1) * TILE_SIZE - 1) / scale);

    // convert the word coordinates to latitude and longitude
    var topLeft = map.getProjection().fromPointToLatLng(topLeftWorldCoord);
    var bottomRight = map.getProjection().fromPointToLatLng(bottomRightWorldCoord);

    // create a rectangle and add it to the map
    var rectangle = new google.maps.Rectangle({
        strokeColor: '#FF0000',
        strokeOpacity: 0.8,
        strokeWeight: 2,
        fillColor: '#FF0000',
        fillOpacity: 0.35,
        map: map,
        clickable: false,
        bounds: {
            north: topLeft.lat(),
            south: bottomRight.lat(),
            east: bottomRight.lng(),
            west: topLeft.lng()
        }
    });
    return rectangle;
}

// Return the filename for the tile with given zoom level and coordinates
function getTileImage(zoom, coord) {
    return ROOT_DIR +
        '/' + getSlideName() +
        '/' + zoom +
        '/' + coord.y +
        '/' + coord.x +
        '.jpg';
}

function getSlideName() {
    if(slideName != '')
        return slideName;

    var urlparameter = '';
    if (window.location.href.indexOf('slide') > -1) {
        urlparameter = getUrlVars()['slide'];
    }
    slideName = urlparameter;
    return slideName;
}

function getMaxZoom() {
    if(maxZoom >= 0)
        return maxZoom;
    
    var url = MAX_ZOOM_ENDPOINT + "?slide=" + getSlideName();
    
    var xhr = new XMLHttpRequest();
    xhr.open('GET', url, false);
    xhr.send();
    
    var zoom = xhr.responseText;
    maxZoom = parseInt(zoom);
    return maxZoom;
}

function getAllSlides() {
    var url = ALL_SLIDES_ENDPOINT;
    
    var xhr = new XMLHttpRequest();
    xhr.open('GET', url, false);
    xhr.send();
    
    var all_slides = JSON.parse(xhr.responseText);
    return all_slides.slides;
}

function getUrlVars() {
    var vars = {};
    var parts = window.location.href.replace(/[?&]+([^=&]+)=([^&]*)/gi, function (m, key, value) {
        vars[key] = value;
    });
    return vars;
}


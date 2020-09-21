<!--This is the home landing screen for the desktop app.
It contains the image icon and (Add Image) button inside the body if the screen is empty.<style scoped>
It contains a progress bar if images are being uploaded, 
and the (Add Image) button will be moved to the button bar area below.
-->

<template>
    <div class="AnalyzePanel">
        <md-dialog-alert
            md-title="Error"
            :md-active.sync="fileError"
            :md-content="fileErrorMsg"
            md-confirm-text="OK" 
        />
        <div v-if="files.length === 0">
            <br>
            <img src="@/assets/image.png" height=100px width=100px />
            <!-- getting rid of duplicate Add Image functionality
            <br>
            <button v-on:click="triggerInput()"> Add Images </button>
            <input type="file" id="file" ref="fileInput" v-on:change="handleFileUpload()" multiple style="display: none" />
            -->
        </div>
        <div v-if="files.length > 0">
            <span class="AnalyzeInstructions">Select files to analyze</span>
            <div class="tableHeaders">
              <span>Index</span>
              <span>File Name</span>
              <span>Size</span>
              <span>Analysis</span>
            </div>
            <div v-for="(file, index) in files" :key="file.name">
                <AnalyzeRow
                    :number="index"
                    :name="file.name"
                    :size="file.sizeFormatted"
                    :analyzed="file.status"
                />
            </div>
            <ProgressBar 
              :message="message"
              :waiting="waiting"
              :processing="processing"
              :progressValue="progressValue"
              :progressBuffer="progressBuffer"
            />
        </div>
    </div>
</template>
<script>
import EventBus from '@/EventBus.js';
import constants from '@/constants.js';
import utils from '@/utils.js';
import AnalyzeRow from '@/components/AnalyzeRow';
import ProgressBar from '@/components/ProgressBar';

export default {
  data() {
    return {
      fileError: false, // Flag if a non-tiff file has been uploaded
      fileErrorMsg: '',
      files: [], // Array ("queue") of files to analyze. TODO: Read/write from user data
      processQueue: [],
      waiting: false, // Currently waiting for something to finish
      processing: false, // Currently processing a file
      tileCount: 0,
      completeCount: 0,
      totalTiles: 0,
      progressValue: 0,
      progressBuffer: 0,
      pythonProcess: null,

      // For progress bar
      message: '',
    };
  },
  components: {
    AnalyzeRow,
    ProgressBar,
  },
  methods: {
    handleFileUpload() {
      EventBus.$emit('onFilesUploaded', this.$refs.fileInput.files);
    },
    triggerInput() {
      const elem = this.$refs.fileInput;
      elem.click();
    },
    // Handle receiving a message from Python and update progress
    handleProgressMessage(message) {
      if (message) {
        let msg = '';
        msg = JSON.parse(message);
        if (msg.initializing) {
          this.tileCount = 0;
          this.completeCount = 0;
          this.progressValue = 0;
          this.progressBuffer = 0;
          this.message = `${constants.INITIALIZING} file '${this.activeFile.name}'`;
          this.waiting = true;
        }
        if (msg.numTiles) {
          this.totalTiles = msg.numTiles;
          this.processing = true;
          this.message = `${constants.PROCESSING} file '${this.activeFile.name}'`;
          this.waiting = false;
        }
        if (msg.processingTile) {
          this.tileCount += 1;
          this.progressBuffer = Math.round((this.tileCount / this.totalTiles) * 100);
        }
        if (msg.completedTile) {
          this.completeCount += 1;
          this.progressValue = Math.round((this.completeCount / this.totalTiles) * 100);
        }
        if (msg.tilingComplete) {
          this.processing = false;
          this.waiting = true;
          this.message = `${constants.HPF} for file '${this.activeFile.name}'`;
        }
        if (msg.processingComplete) {
          // Save results to the persisted store
          this.$store.dispatch('addAnalyzedFile', ({
            filename: this.activeFile.name,
            count: msg.eosinophilCount,
            hpf: msg.hpf,
          }));
          // Remove active file from queue
          this.files = this.files.filter(file => file.name !== this.activeFile.name);
          EventBus.$emit('slideComplete');
        }

        if (msg.terminating) {
          EventBus.$emit('slideComplete');
        }
      }
    },
    analyzeAllSelectedFiles() {
      // queue all selected files and process one at a time
      this.processQueue = this.files.filter(file => file.selected);
      this.processQueue.forEach((file) => {
        file.status = 'Queued';
      });
      this.analyzeFile(this.processQueue[0]);
    },
    analyzeFile(file) {
      const { PythonShell } = require('python-shell');

      const options = {
        mode: 'text',
        pythonOptions: ['-u'], // get print results in real-time
        scriptPath: 'src/py/',
      };

      this.activeFile = file;
      this.activeFile.status = 'Processing';
      options.args = file.path;

      const pyshell = new PythonShell('ProcessSlide.py', options);
      this.pythonProcess = pyshell.childProcess;
      pyshell.on('message', (message) => {
        console.log(message);
        this.handleProgressMessage(message, file);
      });
    },
  },
  created() {
    EventBus.$on('onFilesUploaded', (files) => {
      const userUploadedArray = Array.from(files); // Convert the file oist object-thing to an array
      const existingFileNames = this.files.map(f => f.name);

      // Show an error if an unsupported image file is attempted to be added
      if (userUploadedArray.some(file => constants.SUPPORTED_FILE_EXTENSIONS.indexOf(file.name.split('.').pop()) < 0)) {
        this.fileErrorMsg = constants.FILE_TYPE_ERROR_MSG;
        this.fileError = true;
        return;
      }

      // Show an error if a duplicate file is attempted to be added
      if (userUploadedArray.some(file => existingFileNames.indexOf(file.name) >= 0)) {
        this.fileErrorMsg = constants.DUPLICATE_ERROR_MSG;
        this.fileError = true;
        return;
      }

      this.fileError = false;
      userUploadedArray.forEach((file) => {
        file.sizeFormatted = utils.formatBytes(file.size);
        this.files.push(file);
      });
    });

    EventBus.$on('slideComplete', () => {
      this.processQueue.shift();
      this.waiting = false;
      this.processing = false;
      if (this.processQueue.length === 0) {
        EventBus.$emit('analyzeComplete');
      } else {
        this.analyzeFile(this.processQueue[0]);
      }
    });

    EventBus.$on('onAnalysisRowEnabled', (i) => {
      this.files[i].selected = true;
      EventBus.$emit('onAnalysisReadyStatusChange', true);
    });

    EventBus.$on('onAnalysisRowDisabled', (i) => {
      this.files[i].selected = false;
      EventBus.$emit('onAnalysisReadyStatusChange', this.files.some(file => file.selected));
    });

    EventBus.$on('removeSelectedFromQueue', () => {
      this.files = this.files.filter(file => !file.selected);
      EventBus.$emit('onAnalysisReadyStatusChange', this.files.length > 0);
    });

    EventBus.$on('analyze', () => {
      this.analyzeAllSelectedFiles();
      EventBus.$emit('onAnalysisReadyStatusChange', false);
    });

    EventBus.$on('cancelAnalysis', () => {
      if (this.pythonProcess !== null) {
        this.pythonProcess.kill('SIGINT');
      }
      this.processQueue.forEach((file) => {
        file.status = 'Pending';
      });
      this.processQueue = [];
      EventBus.$emit('onAnalysisReadyStatusChange', this.files.some(file => file.selected));
    });
  },
};
</script>

<style scoped>
  .AnalyzePanel {
    background-color: white;
    height: 400px;
    width: var(--panelWidth);
    margin-top: 30px;
    margin-left: auto;
    margin-right: auto;
    margin-bottom: 30px;
    border-radius: 3px;
    box-shadow: 0px 1px 7px -1px;
  }

  .AnalyzeInstructions {

  }

  .tableHeaders {
    display: grid;
    grid-template-columns: 60fr 250fr 250fr 140fr;
    margin-left: 30px;
    margin-right: 30px;
    margin-top: 30px;
    height: 30px;
    border-bottom: 1px solid var(--light-gray);
    font-weight: bold;
  }

  button {
    --height: 40px;
    height: var(--height);
    width: 100px;
    border: none;
    border-radius: calc( var(--height) / 2 );
    background-color: white;
    color: var(--blue);
    box-shadow: 0px 1px 7px -1px;
    outline: none;
  }
  img {
    margin-top: 50px;
  }
</style>
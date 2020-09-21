<!--This is where the bottom button navigation lies.<style scoped>
Depending on the which tab(analyze/results) you're in, the bottom buttons differ.<style scoped>
It will be either (Analyze) / (Add Images)(Analyze) / (Prev)(Next)(Done)
v-if & v-else are if/else statements-->
<template>
    <div class="ButtonsBar">
      <md-dialog-confirm
        :md-active.sync="showConfirmation"
        md-title="Confirm"
        :md-content="warningMsg"
        md-confirm-text="Yes"
        md-cancel-text="No"
        @md-confirm="onConfirmToClear"
      />
        <div>
            <input type="file" id="file" ref="fileInput" v-on:change="handleFileUpload()" multiple style="display: none" />
            <button class="whiteButton" v-if="currentTab === ANALYZE" v-on:click="triggerInput()">Add Images</button>
        </div>
        <div>
            <button class="whiteButton" v-if="currentTab === ANALYZE && showCancel" v-on:click="onCancelAnalysis()">Stop</button>
            <button class="whiteButton" v-if="currentTab === ANALYZE && showDeleteAnalyze" v-on:click="onRemove()">Remove</button>
            <button v-if="currentTab === ANALYZE" :class="{ isDisabled: !analyzeReady }" v-on:click="onAnalyze()">
              Analyze
            </button>
            <button class="whiteButton" v-if="currentTab === RESULTS && showDeleteResults" v-on:click="showWarningAllFiles()">Remove All</button>
            <button v-if="currentTab === RESULTS && showDeleteResults" v-on:click="showWarningFiles()">Remove Selected</button>
        </div>
    </div>
</template>
<script>
import EventBus from '@/EventBus.js';
import constants from '@/constants.js';

export default {
  props: {
    currentTab: {
      default: constants.ANALYZE_PANEL,
      type: String,
    },
    analyzeReady: {
      default: false,
      type: Boolean,
    },
    showCancel: {
      default: false,
      type: Boolean,
    },
    showDeleteAnalyze: {
      default: false,
      type: Boolean,
    },
    showDeleteResults: {
      default: false,
      type: Boolean,
    },
  },
  data() {
    return {
      ANALYZE: constants.ANALYZE_PANEL,
      RESULTS: constants.RESULTS_PANEL,
      files: [], // TO CONSIDER: I don't like that this is a duplicate of AnalyzePanel.vue
      showConfirmation: false,
      warningMsg: constants.WARNING_MESSAGE_REMOVE_FILE,
    };
  },
  methods: {
    triggerInput() { // Hack to disguise button as html input
      const elem = this.$refs.fileInput;
      elem.click();
    },
    handleFileUpload() {
      EventBus.$emit('onFilesUploaded', this.$refs.fileInput.files);
    },
    onAnalyze() {
      if (this.analyzeReady) {
        EventBus.$emit('analyze');
      }
      // TODO: Consider showing a tooltip or something to explain why this is disabled
    },
    onRemove() {
      EventBus.$emit('removeSelectedFromQueue');
    },
    onCancelAnalysis() {
      EventBus.$emit('cancelAnalysis');
    },
    // onBack() { // Go back to analyze page
    //   EventBus.$emit('setPanel', constants.ANALYZE_PANEL);
    // },
    showWarningFiles() {
      this.warningMsg = constants.WARNING_MESSAGE_REMOVE_FILE;
      this.showConfirmation = true;
    },
    showWarningAllFiles() {
      this.warningMsg = constants.WARNING_MESSAGE_REMOVE_ALL;
      this.showConfirmation = true;
    },
    onConfirmToClear() {
      if (this.warningMsg === constants.WARNING_MESSAGE_REMOVE_ALL) {
        this.$store.dispatch('removeAllAnalyzedFiles');
      } else {
        EventBus.$emit('removeSelectedFromResults');
      }

      this.showConfirmation = false;
    },
  },
};
</script>
<style scoped>
  .previous {
    --height: 25px;
    cursor: pointer;
    height: var(--height);
    width: 25px;
    border: none;
    border-radius: none;
    background-color: var(--blue);
    color: white;
    box-shadow: 0px 1px 7px -1px;
    outline: none;
  }
  .next {
    --height: 25px;
    cursor: pointer;
    height: var(--height);
    width: 25px;
    border: none;
    border-radius: none;
    background-color: var(--blue);
    color: white;
    box-shadow: 0px 1px 7px -1px;
    outline: none;
  }

  .ButtonsBar {
    height: 50px;
    width: var(--panelWidth);
    border-radius: 3px;
    font-size: 16px;
    text-align: center;
    display: grid;
    grid-template-columns: 1fr 1fr;
    margin-left: auto;
    margin-right: auto;
  }

  button {
    --height: 40px;
    cursor: pointer;
    height: var(--height);
    width: 100px;
    border: none;
    border-radius: calc( var(--height) / 2 );
    background-color: var(--blue);
    color: white;
    box-shadow: 0px 1px 7px -1px;
    outline: none;
  }

  .isDisabled {
    background-color: var(--gray);
    cursor: not-allowed;
  }

  .whiteButton {
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

</style>

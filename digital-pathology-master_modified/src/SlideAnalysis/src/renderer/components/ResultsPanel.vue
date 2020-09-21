<!--This page is for writing the functionalities behind viewing the Results 
when the backend completes analyzing the images.
v-for is a loop function-->
<template>
    <div class="ResultsPanel">
      <div v-if="analyzed.length === 0">
        <span>Nothing Analyzed</span>
        <br>
      </div>
      <div v-if="analyzed.length > 0" class="tableHeaders">
          <span>File Name</span>
          <span>Date/Time</span>
          <span>Eosinophil Count</span>
          <span style="justify-self: end;">High Power Field</span>
      </div>
      <ul>
          <li v-for="(file, index) in analyzed" :key="index">
              <ResultRow
                :number="index"
                :name="file.filename"
                :count="file.count"
              />
          </li>
      </ul>
  </div>
</template>

<script>
import EventBus from '@/EventBus.js';
import ResultRow from '@/components/ResultRow';

export default {
  components: {
    ResultRow,
  },
  computed: {
    analyzed() {
      return this.$store.getters.getAnalyzedFiles;
    },
  },
  created() {
    EventBus.$on('onResultRowEnabled', (i) => {
      this.$store.dispatch('toggleSelectedFile', i, true);
      EventBus.$emit('onResultsReadyStatusChange', true);
    });

    EventBus.$on('onResultRowDisabled', (i) => {
      this.$store.dispatch('toggleSelectedFile', i, false);
      EventBus.$emit('onResultsReadyStatusChange', this.analyzed.some(file => file.selected));
    });
  },
};
</script>

<style scoped>
    .ResultsPanel {
        background-color: white;
        height: 400px;
        width: var(--panelWidth);
        margin-top: 30px;
        margin-left: auto;
        margin-right: auto;
        margin-bottom: 30px;
        border-radius: 3px;
        box-shadow: 0px 1px 7px -1px;
        overflow-y: scroll;
    }
    .tableHeaders {
        display: grid;
        grid-template-columns: 1fr 1fr 1fr 1fr 1fr;
        margin-left: 30px;
        margin-top: 30px;
        margin-right: -100px;
        height: 30px;
        border-bottom: 1px solid var(--light-gray);
        font-weight: bold;
    }
    ul {
        list-style: none;
        padding-left: 0%;
    }
    .ResultsPanel::-webkit-scrollbar-thumb {
        background-color: blue;
    }
</style>

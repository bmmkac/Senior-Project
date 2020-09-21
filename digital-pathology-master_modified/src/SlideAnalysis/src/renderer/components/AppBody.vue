<!--This page deals with the entire desktop body area.
v-if & v-else are if/else statements controlling the view 
of the body area depending on which tab is clicked-->
<template>
    <div class="AppBody">
        <Tabs :currentTab="currentTab" />
        <AnalyzePanel v-show="currentTab === ANALYZE" />
        <ResultsPanel v-show="currentTab === RESULTS" />
        <ButtonsBar :currentTab="currentTab" :analyzeReady="analyzeReady" :showCancel="analysisActive" :showDeleteAnalyze="analyzeReady" :showDeleteResults="resultRemoveReady" />
    </div>
</template>

<script>
import Tabs from '@/components/Tabs';
import AnalyzePanel from '@/components/AnalyzePanel';
import ResultsPanel from '@/components/ResultsPanel';
import ButtonsBar from '@/components/ButtonsBar';
import EventBus from '@/EventBus.js';
import constants from '@/constants.js';

export default {
  data() {
    return {
      ANALYZE: constants.ANALYZE_PANEL,
      RESULTS: constants.RESULTS_PANEL,
      currentTab: constants.ANALYZE_PANEL, // Sets the default view
      analyzeReady: false, // Enables/disable analyze button in buttons bar
      analysisActive: false, // Flag for if analysis is running
      resultRemoveReady: false, // Flag if a file is selected to remove in results page
    };
  },
  components: {
    Tabs,
    AnalyzePanel,
    ResultsPanel,
    ButtonsBar,
  },
  created() {
    EventBus.$on('setPanel', (panel) => {
      this.currentTab = panel;
    });
    EventBus.$on('onAnalysisReadyStatusChange', (status) => {
      this.analyzeReady = status;
    });
    EventBus.$on('onResultsReadyStatusChange', (status) => {
      this.resultRemoveReady = status;
    });
    EventBus.$on('analyze', () => {
      this.analysisActive = true;
    });
    EventBus.$on('analyzeComplete', () => {
      this.analysisActive = false;
      this.analyzeReady = false;
    });
    EventBus.$on('cancelAnalysis', () => {
      this.analysisActive = false;
      this.analyzeReady = false;
    });
  },
};
</script>

<style scoped>
.AppBody {
  background-color: var(--light-gray);
  padding-top: 30px;
  padding-bottom: 30px;
  font-size: 16px;
}
</style>
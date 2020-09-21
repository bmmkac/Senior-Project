<!--This page is for the Analyze/Results Tab functionalties & styles.
@click is onclick.
:class is binding a class to manipulate an element’s class list and its inline styles.-->
<template>
    <div class="Tabs">
      <div @click="activateAnalyze" class="TabButton" :class="{ isActive: currentTab === ANALYZE }">
        Analyze
      </div>
      <div @click="activateSaved" class="TabButton" :class="{ isActive: currentTab === RESULTS }">
        Results
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
  },
  data() {
    return {
      ANALYZE: constants.ANALYZE_PANEL,
      RESULTS: constants.RESULTS_PANEL,
    };
  },
  methods: {
    activateAnalyze() {
      EventBus.$emit('setPanel', constants.ANALYZE_PANEL);
    },
    activateSaved() {
      EventBus.$emit('setPanel', constants.RESULTS_PANEL);
    },
  },
};
</script>

<!--here is where the CSS goes-->
<style scoped>
    .Tabs {
      height: 50px;
      width: var(--panelWidth);
      border-radius: 3px;
      background-color: white;
      font-size: 16px;
      text-align: center;
      display: grid;
      grid-template-columns: 1fr 1fr;
      cursor: pointer;
      margin-left: auto;
      margin-right: auto;
    }

    .TabButton {
        line-height: 50px;
        box-shadow: 0px 1px 7px -1px;
        border-radius: 3px;
    }

    .isActive {
        border-bottom: 3px solid var(--blue);
    }

</style>

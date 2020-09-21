<template>
    <div class="AnalyzeRow" :class="{ isActive: active }" v-on:click="onRowSelected()">
        <div>{{ number }}</div>
        <div>{{ name }}</div>
        <div>{{ size }}</div>
        <div>{{ analyzed }}</div>
    </div>
</template>

<script>
import EventBus from '@/EventBus.js';

export default {
  props: {
    number: {
      default: 1,
      type: Number,
    },
    name: {
      default: 'slide1.tiff',
      type: String,
    },
    size: {
      default: 'N/A',
      type: String,
    },
    analyzed: {
      default: 'Pending',
      type: String,
    },
  },
  methods: {
    onRowSelected() {
      if (this.active) {
        this.active = false;
        EventBus.$emit('onAnalysisRowDisabled', this.number);
        return;
      }
      this.active = true;
      EventBus.$emit('onAnalysisRowEnabled', this.number);
    },
  },
  data() {
    return {
      active: false,
    };
  },
};
</script>

<style scoped>
  td {
    border: 1px solid #cccccc;
  }

.AnalyzeRow {
  cursor: pointer;
  display: grid;
  grid-template-columns: 60fr 250fr 250fr 140fr;
  margin-left: 30px;
  margin-right: 30px;
  margin-top: 13px;
  margin-bottom: 10px;
  border-bottom: 1px solid var(--light-gray);
  user-select: none;
}

  .isActive {
    background-color: var(--blue);
    color: white;
  }
</style>

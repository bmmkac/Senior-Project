<template>
<div v-if="processing || waiting">
  <span>{{ message }}</span>
    <div v-if="processing">
        <md-progress-bar md-mode="buffer" :md-value="progressValue" :md-buffer="progressBuffer" />
    </div>
    <div v-if="waiting">
        <md-progress-bar md-mode="query" />
    </div>
</div>
</template>

<script>
export default {
  props: {
    message: {
      default: 'Initializing...',
      type: String,
    },
    processing: {
      default: false,
      type: Boolean,
    },
    waiting: {
      default: false,
      type: Boolean,
    },
    progressValue: {
      default: 0,
      type: Number,
    },
    progressBuffer: {
      default: 0,
      type: Number,
    },
  },
};
</script>

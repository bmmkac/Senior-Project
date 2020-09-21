<!--This is where you edit each result row's output view and functionality.
You have to edit the eyeIcon pop up here.-->
<template>
    <div class="ResultRow" :class="{ isActive: active }" @click="onRowSelected()">
        <div class="filename">{{ name }}</div>
        <div class="datetime">{{ datetime }}</div>
        <div class="count">{{ count }}</div>
        <div class="eyebtn">
            <fa-icon icon="eye" @click.stop="onEyeClicked()"/>
        </div>
    </div>
</template>

<script>
import EventBus from '@/EventBus.js';

export default {
  props: {
    number: {
      default: -1,
      type: Number,
    },
    name: {
      default: 'eo1.jpg',
      type: String,
    },
    datetime: {
      default: '5/9/2019 6:00pm',
      type: String,
    },
    count: {
      default: 0,
      type: Number,
    },
  },
  methods: {
    onEyeClicked() {
      if (this.number > -1) {
        EventBus.$emit('onImagePopup', this.number);
      }
    },
    onRowSelected() {
      if (this.active) {
        this.active = false;
        EventBus.$emit('onResultRowDisabled', this.number);
        return;
      }
      EventBus.$emit('onResultRowEnabled', this.number);
      this.active = true;
    },
  },
  data() {
    return {
      active: false,
      clickedEye: false,
    };
  },
  created() {
    EventBus.$on('removeSelectedFromResults', () => {
      this.$store.dispatch('removeAnalyzedFile', this.number);
      if (this.$store.getters.getAnalyzedFiles.length < 1) {
        EventBus.$emit('onResultsReadyStatusChange', false);
      }
    });
  },
};
</script>

<style scoped>
  td {
      border: 1px solid #cccccc;
  }
  
  .ResultRow {
      display: grid;
      grid-template-columns: 250px 225px 180px 180px 0px;
      margin-top: 13px;
      margin-bottom: 10px;
      border-bottom: 1px solid var(--light-gray);
  }
  .eyebtn {
    cursor: pointer;
    justify-self: end;
  }
  .isActive {
    background-color: var(--blue);
    color: white;
  }
</style>

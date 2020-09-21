
<!--After clicking the eyeIcon popup, 
this is where you edit the functionality of the image modal popup.
@click is onclick anywhere outside of the image screen, the popup closes.-->
<template>
    <div class="ImageModal background" @click="onPopupClose">
        <div class="body">
            <img v-bind:src="imgFileLocation">
        </div>
    </div>
</template>

<script>
export default {
  props: {
    imgFileLocation: {
      default: '', // TODO: Make default some sort of "not found" icon
      type: String,
    },
  },
  methods: {
    onPopupClose() {
      this.$emit('onPopupClose');
    },
  },
};
</script>

<style scoped>
  .ImageModal {
    position: fixed;
    width: 100vw;
    height: 100vh;
    top: 0;
    left: 0;
    display: grid;
    align-items: center;
    justify-content: center;
  }

  .background {
    background: rgba(0, 0, 0, 0.7);
  }

  .body {
    position: relative;
    background-color: white;
    border-radius: 5px;
    overflow: scroll;
    width: 500px;
    height: 500px;
  }

</style>

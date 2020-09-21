<template>
  <div id='app'>
    <AppHeader />
    <AppBody />
    <ImageModal v-if='showImageModal'
      @onPopupClose='hideImageModal'
      :imgFileLocation='imgFileLocation'
    />
  </div>
</template>

<script>
import AppHeader from '@/components/AppHeader';
import AppBody from '@/components/AppBody';
import ImageModal from '@/components/ImageModal';
import EventBus from '@/EventBus.js';

export default {
  components: {
    AppHeader,
    AppBody,
    ImageModal,
  },
  data() {
    return {
      showImageModal: false,
      imgFileLocation: '',
    };
  },
  methods: {
    hideImageModal() {
      this.showImageModal = false;
    },
  },
  created() {
    EventBus.$on('onImagePopup', (number) => {
      this.imgFileLocation = this.$store.getters.getAnalyzedFiles[number].hpf.slide_path;
      this.showImageModal = true;
    });
  },
};
</script>
<style>
@import url('https://fonts.googleapis.com/css?family=Roboto');
:root {
  --blue: #011F5B;
  --red: #990000;
  --gray: #44464B;
  --dark-gray: #131315;
  --light-gray: #f2f2f3;
  --green: #008200;
  --panelWidth: 700px;
}
* { 
  font-family: "Roboto", "Open Sans", sans-serif;
}
#app {
  text-align: center;
}
body {
  margin: 0;
}
</style>

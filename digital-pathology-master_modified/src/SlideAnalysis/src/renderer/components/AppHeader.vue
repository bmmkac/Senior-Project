<!--This page holds the header which contains the project icon, the title "Digital Pathology" 
and the EXIT button to close the desktop app.-->
<template>
    <div class="AppHeader">
        <img src="@/assets/logo.png" width=69px height=69px/>
        <h1> Digital Pathology </h1>
        <button @click="close"> Exit </button>
    </div>
</template>
<script>
    import EventBus from '@/EventBus.js';
    const { remote } = require('electron');

    export default {
      data() {
        return {
          w: remote.getCurrentWindow(),
        };
      },
      methods: {
        close() {
          EventBus.$emit('cancelAnalysis');
          this.w.close();
        },
      },
    };
</script>
<style scoped>
    .AppHeader { 
        background-color: var(--blue);
        height: 100px;
        display: grid;
        grid-template-columns: 100px auto 100px;
        align-items:center;
    }
    img {
        height: 69px;
        width: 69px;
        align-self: center;
    }
    button {
        --height: 40px;
        height: var(--height);
        width: 84px;
        border: none;
        border-radius: calc( var(--height) / 2 );
        background-color: white;
        box-shadow: 0px 1px 7px -1px;
        outline: none;
    }
    h1{
        display: grid;
        font-size: 30px;
        text-align: left;
        vertical-align: bottom;
        align-content: flex-end;
        color: white;
    }
</style>
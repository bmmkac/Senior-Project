import Vue from 'vue';
import axios from 'axios';
import { library } from '@fortawesome/fontawesome-svg-core';
import { faEye } from '@fortawesome/free-solid-svg-icons';
import { FontAwesomeIcon } from '@fortawesome/vue-fontawesome';

// Import the entire VueMaterial for progress bar and its css
// Iporting just the MdProgressBar doesn't seem to work
import VueMaterial from 'vue-material';
import 'vue-material/dist/vue-material.min.css';
import 'vue-material/dist/theme/default.css';

import App from './App';
import router from './router';
import store from './store';

// Add FontAwesome stuff
library.add(faEye);
Vue.component('fa-icon', FontAwesomeIcon);

// For backend
const { PythonShell } = require('python-shell');

if (!process.env.IS_WEB) Vue.use(require('vue-electron'));
Vue.http = Vue.prototype.$http = axios;
Vue.config.productionTip = false;
Vue.prototype.$shell = PythonShell; // For global access of python code by invoking 'this.$shell'

// For 3rd party progress bar
Vue.use(VueMaterial);

/* eslint-disable no-new */
new Vue({
  components: { App },
  router,
  store,
  template: '<App/>',
}).$mount('#app');

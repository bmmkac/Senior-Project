const state = {
  analyzedFiles: [],
};

const mutations = {
  ADD_ANALYZED_FILE(state, { filename, count, hpf }) {
    state.analyzedFiles.push({ filename, count, hpf });
  },
  TOGGLE_SELECTED_FILE(state, index, toggle) {
    state.analyzedFiles[index].selected = toggle;
  },
  REMOVE_ANALYZED_FILE(state, index) {
    state.analyzedFiles.splice(index, 1);
  },
  REMOVE_ALL_ANALYZED_FILES(state) {
    state.analyzedFiles = [];
  },
};

const actions = {
  addAnalyzedFile({ commit }, { filename, count, hpf }) {
    commit('ADD_ANALYZED_FILE', { filename, count, hpf });
  },
  toggleSelectedFile({ commit }, index, toggle) {
    commit('TOGGLE_SELECTED_FILE', index, toggle);
  },
  removeAnalyzedFile({ commit }, index) {
    commit('REMOVE_ANALYZED_FILE', index);
  },
  removeAllAnalyzedFiles({ commit }) {
    commit('REMOVE_ALL_ANALYZED_FILES');
  },
};

const getters = {
  getAnalyzedFiles(state) {
    return state.analyzedFiles;
  },
};

export default {
  state,
  mutations,
  actions,
  getters,
};

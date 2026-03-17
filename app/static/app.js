// Global locale (injected by Jinja2 as window.APP_LANGUAGE)
let _locale = {};

async function loadLocale() {
  const lang = window.APP_LANGUAGE || 'en';
  const r = await fetch(`/api/locale?lang=${lang}`);
  _locale = await r.json();
}

function t(key) {
  return _locale[key] || key;
}

function app() {
  return {
    // Navigation
    currentSection: 'transcribe',

    // Providers
    providers: [],
    selectedProvider: null,
    selectedModel: null,

    // Transcribe section
    files: [],          // { name, size, status, progress, jobId, outputFiles, error }
    outputFormats: ['txt', 'srt'],
    mergeOutput: false,
    prompt: '',
    speakerLabels: false,
    isRunning: false,

    // Settings
    settings: {},

    // History
    historyItems: [],

    // Wizard
    showWizard: !window.APP_WIZARD_COMPLETE,

    // WebSocket
    _ws: null,

    async init() {
      await loadLocale();
      await this.loadProviders();
      await this.loadSettings();
      await this.loadHistory();
      this._connectWs();
    },

    async loadProviders() {
      const r = await fetch('/api/providers');
      this.providers = await r.json();
      if (this.providers.length) {
        this.selectedProvider = this.providers[0];
        if (this.selectedProvider.models.length)
          this.selectedModel = this.selectedProvider.models[0];
      }
    },

    async loadSettings() {
      const r = await fetch('/api/settings');
      this.settings = await r.json();
      this.outputFormats = this.settings.default_output_formats || ['txt'];
      // Set provider/model from settings
      const sp = this.providers.find(p => p.name === this.settings.default_provider);
      if (sp) {
        this.selectedProvider = sp;
        const sm = sp.models.find(m => m.id === this.settings.default_model);
        if (sm) this.selectedModel = sm;
      }
    },

    async loadHistory() {
      const r = await fetch('/api/history');
      this.historyItems = await r.json();
    },

    async saveSettings(patch) {
      await fetch('/api/settings', {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(patch),
      });
      await this.loadSettings();
    },

    handleDrop(event) {
      event.preventDefault();
      const dropped = Array.from(event.dataTransfer.files);
      this._addFiles(dropped);
    },

    handleFileInput(event) {
      this._addFiles(Array.from(event.target.files));
    },

    _addFiles(fileList) {
      const supported = ['mp3','m4a','wav','ogg','flac','webm'];
      for (const f of fileList) {
        const ext = f.name.split('.').pop().toLowerCase();
        if (supported.includes(ext)) {
          this.files.push({ name: f.name, size: f.size, status: 'pending',
                            progress: 0, jobId: null, outputFiles: [], error: null, _file: f });
        }
      }
    },

    removeFile(idx) {
      this.files.splice(idx, 1);
    },

    toggleFormat(fmt) {
      const idx = this.outputFormats.indexOf(fmt);
      if (idx >= 0) this.outputFormats.splice(idx, 1);
      else this.outputFormats.push(fmt);
    },

    async startQueue() {
      if (!this.files.length) return;
      this.isRunning = true;

      const formData = new FormData();
      for (const item of this.files) formData.append('files', item._file);
      formData.append('provider_name', this.selectedProvider?.name || 'openai');
      formData.append('model_id', this.selectedModel?.id || 'whisper-1');
      formData.append('output_formats', JSON.stringify(this.outputFormats));
      formData.append('merge_output', String(this.mergeOutput));
      formData.append('prompt', this.prompt);
      formData.append('speaker_labels', String(this.speakerLabels));

      const r = await fetch('/api/jobs', { method: 'POST', body: formData });
      const { job_id } = await r.json();

      // Mark all files as running (single job for whole queue)
      for (const f of this.files) { f.status = 'running'; f.jobId = job_id; }

      this._subscribeJob(job_id);
    },

    _subscribeJob(jobId) {
      if (!this._ws || this._ws.readyState !== WebSocket.OPEN) return;
      this._ws.send(JSON.stringify({ type: 'subscribe', job_id: jobId }));
    },

    _connectWs() {
      const port = window.APP_PORT || 8000;
      this._ws = new WebSocket(`ws://localhost:${port}/ws`);
      this._ws.onmessage = (e) => {
        const msg = JSON.parse(e.data);
        this._handleWsMessage(msg);
      };
      this._ws.onclose = () => {
        setTimeout(() => this._connectWs(), 2000);
      };
    },

    _handleWsMessage(msg) {
      if (msg.type === 'progress') {
        const item = this.files.find(f => f.jobId === msg.job_id);
        if (item) { item.progress = msg.progress * 100; item.status = 'running'; }
      } else if (msg.type === 'done') {
        for (const f of this.files) {
          if (f.jobId === msg.job_id) {
            f.status = 'done'; f.progress = 100;
            f.outputFiles = msg.output_files || [];
          }
        }
        this.isRunning = false;
        this.loadHistory();
      } else if (msg.type === 'error') {
        const item = this.files.find(f => f.jobId === msg.job_id);
        if (item) { item.status = 'error'; item.error = msg.message; }
        this.isRunning = false;
      }
    },

    stopQueue() {
      const running = this.files.find(f => f.status === 'running');
      if (running?.jobId && this._ws) {
        this._ws.send(JSON.stringify({ type: 'cancel', job_id: running.jobId }));
      }
      this.isRunning = false;
    },

    async deleteHistory(id) {
      await fetch(`/api/history/${id}`, { method: 'DELETE' });
      this.historyItems = this.historyItems.filter(h => h.id !== id);
    },

    async completeWizard(setupType) {
      await this.saveSettings({ wizard_complete: true });
      this.showWizard = false;
    },

    hardwareBadgeClass(hint) {
      const map = {
        cpu: 'bg-green-900 text-green-300',
        cpu_recommended: 'bg-green-900 text-green-300',
        gpu_optional: 'bg-yellow-900 text-yellow-300',
        gpu_recommended: 'bg-orange-900 text-orange-300',
        cloud: 'bg-blue-900 text-blue-300',
      };
      return map[hint] || 'bg-gray-800 text-gray-400';
    },

    formatBytes(bytes) {
      if (bytes < 1024) return bytes + ' B';
      if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
      return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
    },

    t,
  };
}

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

    // ── Install modal state ──────────────────────────────────────
    installModal: null,   // { providerName, providerLabel, packages } or null
    installing: false,
    installProgress: 0,
    installMessage: '',
    installError: null,

    // ── Live mode state ─────────────────────────────────────────
    liveProviders: [],
    liveProvider: null,
    liveModel: null,
    liveSessionId: null,
    liveRecording: false,
    livePaused: false,
    liveTranscript: [],
    liveTimer: 0,
    _liveTimerInterval: null,
    _mediaRecorder: null,
    _analyser: null,
    _animFrame: null,
    ffmpegAvailable: true,
    liveSaveAudio: false,

    // WebSocket
    _ws: null,

    async init() {
      await loadLocale();
      await this.loadProviders();
      await this.loadSettings();
      await this.loadHistory();
      this._connectWs();
      await this._checkFfmpeg();
      this._updateLiveProviders();
    },

    async loadProviders() {
      const r = await fetch('/api/providers');
      this.providers = await r.json();
      if (this.providers.length) {
        this.selectedProvider = this.providers[0];
        if (this.selectedProvider.models.length)
          this.selectedModel = this.selectedProvider.models[0];
      }
      this._updateLiveProviders();
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
      } else if (msg.type === 'install_progress') {
        this.installProgress = msg.progress;
        this.installMessage = msg.message;
      } else if (msg.type === 'install_done') {
        this.installing = false;
        if (msg.success) {
          this.installProgress = 1;
          this.installMessage = this.t('install_done_ok');
          setTimeout(() => {
            this.closeInstallModal();
            this.loadProviders();
          }, 1500);
        } else {
          this.installError = msg.error || this.t('install_done_error');
        }
      } else if (msg.type === 'live_session_started') {
        this.liveSessionId = msg.session_id;

      } else if (msg.type === 'segment') {
        this.liveTranscript.push({ start: msg.start, end: msg.end, text: msg.text });
        this.$nextTick(() => {
          const el = document.getElementById('live-transcript');
          if (el) el.scrollTop = el.scrollHeight;
        });

      } else if (msg.type === 'live_session_stopped') {
        this.liveRecording = false;
        this.liveSessionId = null;
        this.liveTranscript = [];
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

    openInstallModal(provider) {
        this.installModal = {
            providerName: provider.name,
            providerLabel: provider.name.replace(/_/g, '-'),
            packages: provider.name === 'faster_whisper'
                ? 'faster-whisper (~150 MB)'
                : provider.name === 'qwen3_asr'
                    ? 'transformers + torch (~2 GB)'
                    : null,
        };
        this.installing = false;
        this.installProgress = 0;
        this.installMessage = '';
        this.installError = null;
    },

    closeInstallModal() {
        this.installModal = null;
    },

    async confirmInstall() {
        if (!this.installModal) return;
        this.installing = true;
        this.installProgress = 0;
        this.installError = null;
        try {
            const r = await fetch(`/api/install/${this.installModal.providerName}`, { method: 'POST' });
            if (!r.ok) throw new Error(`HTTP ${r.status}`);
        } catch (e) {
            this.installing = false;
            this.installError = e.message;
        }
    },

    selectModel(provider, model) {
        if (!provider.available) {
            this.openInstallModal(provider);
            return;
        }
        this.selectedProvider = provider.name;
        this.selectedModel = model.id;
    },

    async _checkFfmpeg() {
      try {
        const r = await fetch('/api/ffmpeg');
        const d = await r.json();
        this.ffmpegAvailable = d.available;
      } catch (_) {
        this.ffmpegAvailable = false;
      }
    },

    _updateLiveProviders() {
      this.liveProviders = this.providers.filter(p =>
        p.available && p.models.some(m => m.supports_live)
      );
      if (this.liveProviders.length) {
        const first = this.liveProviders[0];
        this.liveProvider = first;
        this.liveModel = first.models.find(m => m.supports_live) || null;
      }
    },

    async startRecording() {
      if (!this.liveProvider || !this.liveModel) return;
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        this._setupWaveform(stream);

        this._ws.send(JSON.stringify({
          type: 'start_live',
          provider_name: this.liveProvider.name,
          model_id: this.liveModel.id,
          opts: { language: null, speaker_labels: false, output_formats: ['txt', 'srt'] },
        }));

        this._mediaRecorder = new MediaRecorder(stream, {
          mimeType: 'audio/webm;codecs=opus',
          audioBitsPerSecond: 64000,
        });
        this._mediaRecorder.ondataavailable = async (e) => {
          if (!e.data || e.data.size === 0 || !this.liveSessionId) return;
          const buf = await e.data.arrayBuffer();
          const b64 = btoa(String.fromCharCode(...new Uint8Array(buf)));
          this._ws.send(JSON.stringify({
            type: 'audio_chunk',
            session_id: this.liveSessionId,
            data: b64,
          }));
        };
        this._mediaRecorder.start(3000);

        this.liveRecording = true;
        this.livePaused = false;
        this.liveTimer = 0;
        this._liveTimerInterval = setInterval(() => { this.liveTimer++; }, 1000);

      } catch (err) {
        console.error('Microphone access denied:', err);
      }
    },

    pauseRecording() {
      if (!this._mediaRecorder) return;
      if (this.livePaused) {
        this._mediaRecorder.resume();
        this.livePaused = false;
      } else {
        this._mediaRecorder.pause();
        this.livePaused = true;
      }
    },

    async stopRecording() {
      if (this._mediaRecorder) {
        this._mediaRecorder.stop();
        this._mediaRecorder.stream.getTracks().forEach(t => t.stop());
        this._mediaRecorder = null;
      }
      clearInterval(this._liveTimerInterval);
      this._liveTimerInterval = null;
      cancelAnimationFrame(this._animFrame);
      this._animFrame = null;

      if (this.liveSessionId) {
        this._ws.send(JSON.stringify({
          type: 'stop_live',
          session_id: this.liveSessionId,
        }));
      }
    },

    _setupWaveform(stream) {
      try {
        const ctx = new AudioContext();
        const src = ctx.createMediaStreamSource(stream);
        this._analyser = ctx.createAnalyser();
        this._analyser.fftSize = 256;
        src.connect(this._analyser);
        this._drawWaveform();
      } catch (_) {
        // Web Audio API unavailable — skip waveform
      }
    },

    _drawWaveform() {
      const canvas = document.getElementById('live-waveform');
      if (!canvas || !this._analyser) return;
      const ctx = canvas.getContext('2d');
      const buf = new Uint8Array(this._analyser.frequencyBinCount);
      const draw = () => {
        this._animFrame = requestAnimationFrame(draw);
        this._analyser.getByteFrequencyData(buf);
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = '#6366f1';
        const bw = canvas.width / buf.length;
        buf.forEach((v, i) => {
          const h = (v / 255) * canvas.height;
          ctx.fillRect(i * bw, canvas.height - h, bw - 1, h);
        });
      };
      draw();
    },

    _formatLiveTime(secs) {
      const m = String(Math.floor(secs / 60)).padStart(2, '0');
      const s = String(secs % 60).padStart(2, '0');
      return `${m}:${s}`;
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

(() => {
  const root = document.querySelector('.retrieval-demo');
  if (!root) return;
  const cards = [...root.querySelectorAll('.demo-video')];
  const videos = cards.map(card => card.querySelector('video'));
  const play = root.querySelector('#demo-play');
  const queryButtons = [...root.querySelectorAll('[data-query]')];
  const stageButtons = [...root.querySelectorAll('[data-stage-button]')];
  const progress = root.querySelector('.demo-progress');
  const starts = [0, 4500, 9000, 13500];
  const duration = 20000;
  const examples = [
    { text: 'Water is being poured into a glass.', winner: 0, intervals: [[[4, 14]], [], [[0, 7], [17, 18]]] },
    { text: 'A person is cutting vegetables on a board.', winner: 1, intervals: [[[0, 4], [14, 18]], [[2, 12]], []] },
    { text: 'A person is washing dishes at the sink.', winner: 2, intervals: [[], [[0, 2], [12, 18]], [[7, 17]]] }
  ];
  const boundaries = [
    [[0, 4, 14, 18], [0, 2, 4, 9, 14, 18], [0, 2, 4, 6, 9, 11, 14, 16, 18]],
    [[0, 2, 12, 18], [0, 2, 7, 12, 15, 18], [0, 1, 2, 4, 7, 9, 12, 15, 16, 18]],
    [[0, 7, 17, 18], [0, 3, 7, 12, 17, 18], [0, 3, 5, 7, 10, 12, 15, 17, 18]]
  ];
  let query = 0;
  let stage = 0;
  let elapsed = 0;
  let running = false;
  let lastFrame = 0;
  let frame = 0;
  let loaded = false;
  const ns = 'http://www.w3.org/2000/svg';
  const createSvg = (name, attrs, parent) => {
    const element = document.createElementNS(ns, name);
    Object.entries(attrs).forEach(([key, value]) => element.setAttribute(key, value));
    parent.append(element);
    return element;
  };
  const nodes = [];
  const edges = [];
  const x = time => 46 + time / 18 * 251;
  cards.forEach((card, videoIndex) => {
    const svg = card.querySelector('.demo-graph');
    const edgeLayer = createSvg('g', {}, svg);
    const nodeLayer = createSvg('g', {}, svg);
    const levels = boundaries[videoIndex].map((cuts, level) => {
      const y = 19 + level * 43;
      createSvg('text', { x: 0, y: y + 4 }, nodeLayer).textContent = ['Coarse', 'Mid', 'Fine'][level];
      return cuts.slice(0, -1).map((start, index) => {
        const end = cuts[index + 1];
        const rect = createSvg('rect', { x: x(start) + 1, y: y - 8, width: x(end) - x(start) - 3, height: 16, rx: 2, fill: '#dce5ed', stroke: '#c7d4df', 'stroke-width': .6 }, nodeLayer);
        const node = { start, end, level, videoIndex, rect, cx: (x(start) + x(end)) / 2, cy: y };
        nodes.push(node);
        return node;
      });
    });
    levels.slice(0, -1).forEach((level, index) => level.forEach(a => levels[index + 1].forEach(b => {
      if (Math.min(a.end, b.end) <= Math.max(a.start, b.start)) return;
      const d = `M${a.cx} ${a.cy + 8} L${b.cx} ${b.cy - 8}`;
      const path = createSvg('path', { d, class: 'graph-edge' }, edgeLayer);
      const pulse = createSvg('circle', { r: 1.9, class: 'graph-pulse' }, edgeLayer);
      createSvg('animateMotion', { dur: '1.8s', repeatCount: 'indefinite', path: d, begin: `${index * .3}s` }, pulse);
      edges.push({ a, b, path, pulse });
    })));
    const chart = card.querySelector('.demo-consensus svg');
    createSvg('line', { x1: 0, x2: 300, y1: 54, y2: 54, stroke: '#e0e8ee' }, chart);
    createSvg('path', { class: 'consensus-fill' }, chart);
    createSvg('path', { class: 'consensus-line', pathLength: 400 }, chart);
  });
  function strength(node) {
    const intervals = examples[query].intervals[node.videoIndex];
    const overlap = intervals.reduce((sum, [a, b]) => sum + Math.max(0, Math.min(node.end, b) - Math.max(node.start, a)), 0);
    return overlap / (node.end - node.start) * (node.videoIndex === examples[query].winner ? 1 : .48);
  }
  function renderEvidence() {
    nodes.forEach(node => {
      const value = strength(node);
      node.rect.setAttribute('fill', stage > 0 && value > 0 ? '#367fae' : '#dce5ed');
      node.rect.setAttribute('fill-opacity', stage > 0 && value > 0 ? .16 + value * .74 : 1);
      node.rect.setAttribute('stroke', stage > 0 && value > .7 ? '#397cae' : '#c7d4df');
    });
    edges.forEach(({ a, b, path, pulse }) => {
      const value = Math.min(strength(a), strength(b));
      path.style.opacity = stage > 0 ? .08 + value * .65 : 0;
      pulse.style.display = stage === 1 && value > .4 ? '' : 'none';
    });
    cards.forEach((card, index) => {
      const intervals = examples[query].intervals[index];
      card.querySelectorAll('.demo-interval').forEach(element => element.remove());
      intervals.forEach(([a, b]) => {
        const interval = document.createElement('span');
        interval.className = 'demo-interval';
        interval.setAttribute('aria-hidden', 'true');
        interval.style.setProperty('--interval-left', `${a / 18 * 100}%`);
        interval.style.setProperty('--interval-width', `${(b - a) / 18 * 100}%`);
        card.querySelector('.demo-filmstrip').append(interval);
      });
      card.classList.toggle('is-selected', stage === 3 && index === examples[query].winner);
      const values = Array.from({ length: 81 }, (_, point) => {
        const t = point / 80 * 18;
        const evidence = intervals.reduce((sum, [a, b]) => {
          const mid = (a + b) / 2;
          const width = Math.max(.7, (b - a) / 3.4);
          return sum + Math.exp(-(((t - mid) / width) ** 2));
        }, 0);
        return `${point ? 'L' : 'M'}${point / 80 * 300} ${(52 - evidence * (index === examples[query].winner ? 41 : 19) - Math.sin(point * .9) ** 2 * 1.6).toFixed(2)}`;
      }).join(' ');
      card.querySelector('.consensus-line').setAttribute('d', values);
      card.querySelector('.consensus-fill').setAttribute('d', values + ' L300 54 H0 Z');
    });
  }
  function loadMedia() {
    if (loaded) return;
    loaded = true;
    videos.forEach(video => { video.preload = 'auto'; video.load(); });
  }
  function seek(video, time) {
    if (video.readyState >= 1) video.currentTime = Math.min(time, video.duration || 18);
    else video.addEventListener('loadedmetadata', () => { video.currentTime = time; }, { once: true });
  }
  function startMedia() {
    videos.forEach((video, index) => {
      if (stage < 3 || index === examples[query].winner) video.play().catch(() => {});
      else video.pause();
    });
  }
  function stop() {
    running = false;
    cancelAnimationFrame(frame);
    videos.forEach(video => video.pause());
    play.textContent = elapsed >= duration ? 'Replay walkthrough' : elapsed ? 'Resume walkthrough' : 'Play walkthrough';
    root.classList.remove('is-playing');
  }
  function setStage(next) {
    stage = next;
    root.dataset.stage = next;
    stageButtons.forEach((button, index) => {
      button.setAttribute('aria-pressed', String(index === next));
      button.classList.toggle('is-complete', index < next);
    });
    const winner = examples[query].winner;
    const interval = examples[query].intervals[winner][0];
    root.querySelector('#demo-explanation').textContent = [
      'Split each video into segments at multiple temporal scales.',
      'Let temporally and semantically related segments reinforce relevant evidence.',
      'Align evidence in time and combine support across scales.',
      `Illustrative selection: Video 0${winner + 1}. Relevant evidence spans ${interval[0]}–${interval[1]} s, while retrieval returns the whole video.`
    ][next];
    renderEvidence();
    if (next === 3) {
      seek(videos[winner], interval[0]);
      videos.forEach((video, index) => { if (index !== winner) video.pause(); });
    }
    if (running) startMedia();
  }
  function updateProgress() {
    const percent = Math.min(100, elapsed / duration * 100);
    progress.firstElementChild.style.width = `${percent}%`;
    progress.setAttribute('aria-valuenow', Math.round(percent));
  }
  function tick(now) {
    if (!running) return;
    elapsed += Math.min(now - lastFrame, 100);
    lastFrame = now;
    const next = starts.reduce((current, start, index) => elapsed >= start ? index : current, 0);
    if (next !== stage) setStage(next);
    updateProgress();
    if (elapsed >= duration) { stop(); return; }
    frame = requestAnimationFrame(tick);
  }
  function reset() {
    stop();
    elapsed = 0;
    setStage(0);
    videos.forEach(video => { if (video.readyState >= 1) video.currentTime = 0; });
    updateProgress();
    play.textContent = 'Play walkthrough';
  }
  function start() {
    if (elapsed >= duration) reset();
    loadMedia();
    running = true;
    root.classList.add('is-playing');
    play.textContent = 'Pause walkthrough';
    lastFrame = performance.now();
    startMedia();
    frame = requestAnimationFrame(tick);
  }
  play.addEventListener('click', () => running ? stop() : start());
  root.querySelector('#demo-reset').addEventListener('click', reset);
  queryButtons.forEach(button => button.addEventListener('click', () => {
    reset();
    query = Number(button.dataset.query);
    queryButtons.forEach(item => item.setAttribute('aria-pressed', String(item === button)));
    root.querySelector('#demo-query-text').textContent = examples[query].text;
    setStage(0);
    if (!matchMedia('(prefers-reduced-motion: reduce)').matches) start();
  }));
  stageButtons.forEach(button => button.addEventListener('click', () => {
    stop();
    loadMedia();
    elapsed = starts[Number(button.dataset.stageButton)];
    setStage(Number(button.dataset.stageButton));
    updateProgress();
    play.textContent = elapsed ? 'Resume walkthrough' : 'Play walkthrough';
  }));
  videos.forEach((video, index) => {
    video.muted = true;
    video.addEventListener('error', () => { cards[index].querySelector('.demo-media-error').hidden = false; });
    video.querySelector('source').addEventListener('error', () => { cards[index].querySelector('.demo-media-error').hidden = false; });
    video.addEventListener('timeupdate', () => {
      cards[index].querySelector('.demo-playhead').style.left = `${Math.min(100, video.currentTime / 18 * 100)}%`;
      if (running && stage === 3 && index === examples[query].winner) {
        const [a, b] = examples[query].intervals[index][0];
        if (video.currentTime >= b) video.currentTime = a;
      }
    });
  });
  document.addEventListener('visibilitychange', () => { if (document.hidden && running) stop(); });
  const observer = new IntersectionObserver(entries => {
    entries.forEach(entry => { if (entry.isIntersecting) loadMedia(); else if (running) stop(); });
  }, { threshold: .05 });
  observer.observe(root);
  setStage(0);
})();

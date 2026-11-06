const data = JSON.parse(document.getElementById('dashboard-data').textContent);
const byId = id => document.getElementById(id);
const escapeText = value => String(value ?? '').replace(/[&<>"']/g, character => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[character]));
const number = (value, digits = 2) => value == null ? 'Unavailable' : Number(value).toLocaleString(undefined, {maximumFractionDigits: digits});
const percent = value => value == null ? 'Unavailable' : `${(100 * value).toFixed(1)}%`;
const comparison = value => value == null ? 'Unavailable' : String(value).replace('At Average', 'Average');
const palette = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f', '#e5c494', '#b3b3b3'];

const overallLookup = new Map(data.overall_nodes.map(node => [node.target_islander, node]));
const lifecycleLookup = new Map(data.lifecycle.map(row => [row.islander, row]));
const episodes = [...new Set(data.episode_nodes.map(node => node.episode))].sort((a, b) => a - b);
const config = {responsive: true, displaylogo: false, scrollZoom: true, modeBarButtonsToRemove: ['select2d', 'lasso2d']};

let selectedName = overallLookup.has('Mackenzie Annis') ? 'Mackenzie Annis' : data.overall_nodes[0].target_islander;
let currentNodes = [], currentEdges = [], shownEdges = [];
let selectedEdge = null;
let networkReady = false;

// Setup islander dropdown
byId('islander').innerHTML = [...overallLookup.keys()].sort().map(name => `<option>${escapeText(name)}</option>`).join('');
byId('islander').value = selectedName;

// --- STEPPER LOGIC FOR EPISODES ---
const viewStates = ['overall', ...episodes];
let currentViewIndex = 0;

function updateViewText() { 
    byId('view-text').textContent = viewStates[currentViewIndex] === 'overall' ? 'Overall' : `Episode ${viewStates[currentViewIndex]}`;
    byId('view-prev').disabled = currentViewIndex === 0;
    byId('view-next').disabled = currentViewIndex === viewStates.length - 1; 
}

function episodeNumber() { 
    return viewStates[currentViewIndex] === 'overall' ? null : viewStates[currentViewIndex]; 
}

byId('view-prev').addEventListener('click', () => { 
    if (currentViewIndex > 0) { 
        currentViewIndex--; 
        updateViewText(); 
        selectedEdge = null; 
        renderAll(true); 
    }
});

byId('view-next').addEventListener('click', () => { 
    if (currentViewIndex < viewStates.length - 1) { 
        currentViewIndex++; 
        updateViewText(); 
        selectedEdge = null; 
        renderAll(true); 
    }
});
// ----------------------------------

function getView() {
    const episode = episodeNumber();
    let nodes = episode == null ? data.overall_nodes : data.episode_nodes.filter(node => node.episode === episode);
    
    if (episode != null && !byId('departed').checked) {
        nodes = nodes.filter(node => {
            const life = lifecycleLookup.get(node.target_islander);
            return life.entrance_episode <= episode && life.exit_episode >= episode;
        });
    }
    
    const names = new Set(nodes.map(node => node.target_islander));
    let edges = episode == null ? data.overall_edges : data.episode_edges.filter(edge => edge.episode === episode);
    edges = edges.filter(edge => names.has(edge.source) && names.has(edge.target));
    return {nodes, edges};
}

function quantile(values, fraction) {
    if (!values.length) return 0;
    const ordered = [...values].sort((a, b) => a - b);
    const index = (ordered.length - 1) * fraction;
    const lower = Math.floor(index);
    const upper = Math.ceil(index);
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower);
}

function nodeHover(node) {
    return `<b>${escapeText(node.target_islander)}</b><br>Comments: ${number(node.num_comments, 0)}<br>Sentiment: ${comparison(node.sentiment_comparison)} (${number(node.roberta_relative_sentiment)} z)<br>Toxicity: ${comparison(node.toxicity_comparison)} (${number(node.toxicity_mean, 3)})<br>Audience Approval (LLM): ${escapeText(node.audience_approval ?? 'Unclear')}<br>Top emojis: ${escapeText(node.top_emojis ?? 'NA')}<br>Standout emojis: ${escapeText(node.standout_emojis ?? 'NA')}`;
}

function edgeHover(edge) {
    return `<b>${escapeText(edge.source)} ↔ ${escapeText(edge.target)}</b><br>Shared comments: ${number(edge.weight, 0)}<br>Discussion tone: ${escapeText(edge.relative_tone ?? 'Unavailable')} (${number(edge.pair_relative_sentiment)} z)<br>Polarized: ${edge.is_polarized == null ? 'Unknown' : edge.is_polarized ? 'Yes' : 'No'}<br>Click for details`;
}

function bounds(nodes, column) {
    const values = nodes.map(node => node[column]);
    if (!values.length) return [-1, 1];
    const low = Math.min(...values), high = Math.max(...values);
    const pad = Math.max(high - low, 0.1) * 0.25;
    return [low - pad, high + pad];
}

async function renderNetwork(reset = false) {
    const view = getView();
    currentNodes = view.nodes;
    currentEdges = view.edges;
    
    const threshold = quantile(currentEdges.map(edge => edge.weight), Number(byId('density').value));
    shownEdges = currentEdges.filter(edge => edge.weight >= threshold);
    
    let visibleNodes = currentNodes;
    const neighbours = new Set([selectedName]);
    
    currentEdges.forEach(edge => {
        if (edge.source === selectedName) neighbours.add(edge.target);
        if (edge.target === selectedName) neighbours.add(edge.source);
    });
    
    if (byId('focus').checked) {
        visibleNodes = currentNodes.filter(node => neighbours.has(node.target_islander));
        shownEdges = shownEdges.filter(edge => edge.source === selectedName || edge.target === selectedName);
    }
    
    const linesX = [], linesY = [], highlightX = [], highlightY = [], hoverX = [], hoverY = [], hoverText = [], edgeIndexes = [];
    
    shownEdges.forEach((edge, index) => {
        const first = overallLookup.get(edge.source);
        const second = overallLookup.get(edge.target);
        const highlighted = edge.source === selectedName || edge.target === selectedName;
        const xs = highlighted ? highlightX : linesX;
        const ys = highlighted ? highlightY : linesY;
        
        xs.push(first.x, second.x, null);
        ys.push(first.y, second.y, null);
        
        for (let step = 1; step < 16; step++) {
            const fraction = step / 16;
            hoverX.push(first.x + fraction * (second.x - first.x));
            hoverY.push(first.y + fraction * (second.y - first.y));
            hoverText.push(edgeHover(edge));
            edgeIndexes.push(index);
        }
    });
    
    const maxComments = Math.max(...data.overall_nodes.map(node => node.num_comments));
    
    const traces = [
        {x: linesX, y: linesY, mode: 'lines', line: {color: 'rgba(110,130,125,0.22)', width: 1.2}, hoverinfo: 'skip'},
        {x: highlightX, y: highlightY, mode: 'lines', line: {color: 'rgba(40,114,104,0.65)', width: 2}, hoverinfo: 'skip'},
        {x: hoverX, y: hoverY, mode: 'markers', marker: {size: 12, color: 'rgba(0,0,0,0)'}, text: hoverText, customdata: edgeIndexes, hovertemplate: '%{text}<extra></extra>'},
        {
            x: visibleNodes.map(node => node.x), 
            y: visibleNodes.map(node => node.y), 
            mode: 'markers+text', 
            text: visibleNodes.map(node => node.target_islander), 
            textposition: 'top center', 
            textfont: {size: 11, color: '#243c3b'}, 
            customdata: visibleNodes.map(node => node.target_islander), 
            hovertext: visibleNodes.map(nodeHover), 
            hovertemplate: '%{hovertext}<extra></extra>', 
            marker: {
                size: visibleNodes.map(node => 10 + 45 * Math.sqrt(node.num_comments / maxComments)), 
                color: visibleNodes.map(node => palette[(overallLookup.get(node.target_islander).community - 1) % palette.length]), 
                line: {
                    color: visibleNodes.map(node => node.target_islander === selectedName ? '#203f39' : 'white'), 
                    width: visibleNodes.map(node => node.target_islander === selectedName ? 3 : 1)
                }
            }
        }
    ];
    
    const episode = episodeNumber();
    const viewLabel = episode == null ? 'Overall' : `Episode ${episode}`;
    
    byId('view-label').textContent = viewLabel.toUpperCase();
    byId('counts').textContent = `${visibleNodes.length} islanders · ${shownEdges.length} of ${currentEdges.length} relationships`;
    
    const layout = {
        margin: {l: 35, r: 35, t: 35, b: 30}, 
        showlegend: false, 
        paper_bgcolor: 'white', 
        plot_bgcolor: 'white', 
        dragmode: 'pan', 
        hovermode: 'closest', 
        hoverlabel: {bgcolor: 'white', font: {color: '#243c3b', size: 12}}, 
        xaxis: {visible: false, range: bounds(visibleNodes, 'x')}, 
        yaxis: {visible: false, range: bounds(visibleNodes, 'y')}, 
        uirevision: reset ? Date.now() : `${viewLabel}-${byId('focus').checked}-${byId('departed').checked}`
    };
    
    await Plotly.react('network', traces, layout, config);
    
    if (!networkReady) {
        byId('network').on('plotly_click', event => {
            const point = event.points[0];
            if (point.curveNumber === 3) {
                selectedName = point.customdata;
                selectedEdge = null;
                byId('islander').value = selectedName;
                renderAll();
            }
            if (point.curveNumber === 2) {
                selectedEdge = shownEdges[point.customdata];
                renderEdge();
            }
        });
        networkReady = true;
    }
}

function metricLine(label, value) {
    return `<div class="metric-line"><span>${label}</span><b>${value}</b></div>`;
}

function renderSidebar() {
    const episode = episodeNumber();
    const node = currentNodes.find(row => row.target_islander === selectedName);
    byId('selected-name').textContent = selectedName;
    
    const life = lifecycleLookup.get(selectedName);
    byId('selected-scope').textContent = `${episode == null ? 'Overall' : `Episode ${episode}`} · entered ${life.entrance_episode} · final recorded episode ${life.exit_episode}`;
    
    if (node) {
        byId('metrics').innerHTML = `<div class="stats"><div class="stat"><strong>${number(node.num_comments, 0)}</strong><span>Distinct comments</span></div><div class="stat"><strong>${number(node.kalshi_price, 1)}${node.kalshi_price == null ? '' : '¢'}</strong><span>Kalshi saved price</span></div></div>` +
        metricLine('Sentiment', `${comparison(node.sentiment_comparison)}<br>${number(node.roberta_relative_sentiment)} z`) + 
        metricLine('Toxicity', `${comparison(node.toxicity_comparison)}<br>${number(node.toxicity_mean, 3)}`) + 
        metricLine('Audience Approval (LLM)', escapeText(node.audience_approval ?? 'Unclear')) + 
        metricLine('Top emojis', `<span class="emoji">${escapeText(node.top_emojis ?? 'NA')}</span>`) + 
        metricLine('Standout emojis', `<span class="emoji">${escapeText(node.standout_emojis ?? 'NA')}</span>`);
    } else {
        const explanation = episode < life.entrance_episode ? 'This islander has not entered yet.' : episode > life.exit_episode && !byId('departed').checked ? 'This islander is past their final exit. Enable “Include departed” to inspect any later discussion.' : 'No saved discussion metrics for this episode.';
        byId('metrics').innerHTML = `<p class="empty">${explanation} Overall and other episodes remain available.</p>`;
    }
    
    const connections = currentEdges.filter(edge => edge.source === selectedName || edge.target === selectedName).sort((a, b) => b.weight - a.weight || a.source.localeCompare(b.source));
    
    byId('connections').innerHTML = connections.length ? connections.map((edge, index) => {
        const name = edge.source === selectedName ? edge.target : edge.source;
        return `<button class="connection" data-connection="${index}"><span>${escapeText(name)}</span><b>${number(edge.weight, 0)}</b></button>`;
    }).join('') : '<p class="empty">No connections in this view.</p>';
    
    byId('connections').querySelectorAll('button').forEach(button => button.addEventListener('click', () => {
        selectedEdge = connections[Number(button.dataset.connection)];
        renderEdge();
    }));
    
    renderEdge();
    renderTrend();
    renderModelCards();
}

function renderEdge() {
    if (!selectedEdge) {
        byId('edge-detail').innerHTML = '';
        return;
    }
    const edge = selectedEdge;
    byId('edge-detail').innerHTML = `<div class="edge-box"><b>${escapeText(edge.source)} ↔ ${escapeText(edge.target)}</b><p>${number(edge.weight, 0)} shared comments · ${number(edge.scored_comments, 0)} scored</p><p>Tone: ${escapeText(edge.relative_tone ?? 'Unavailable')} (${number(edge.pair_relative_sentiment)} z)</p><p>Polarized: ${edge.is_polarized == null ? 'Unknown' : edge.is_polarized ? 'Yes' : 'No'}</p><p>Compared with: ${escapeText(edge.reference_population ?? 'Unavailable')}</p><p>Positive ${percent(edge.positive_share)} · negative ${percent(edge.negative_share)}</p><p class="muted">Approximation from sentiment in co-mentioned target windows; not a judgment of their relationship.</p></div>`;
}

function smallLayout(yTitle) {
    return {
        margin: {l: 43, r: 12, t: 12, b: 37}, 
        font: {size: 10, color: '#647572'}, 
        paper_bgcolor: 'white', 
        plot_bgcolor: 'white', 
        showlegend: false, 
        xaxis: {title: {text: 'Episode', standoff: 5}, gridcolor: '#edf0ec'}, 
        yaxis: {title: {text: yTitle, standoff: 5}, gridcolor: '#edf0ec'}, 
        hovermode: 'closest'
    };
}

function renderTrend() {
    const metric = byId('trend').value;
    const rows = data.episode_nodes.filter(row => row.target_islander === selectedName).sort((a, b) => a.episode - b.episode);
    let x = rows.map(row => row.episode), y, title, note;
    
    if (metric === 'sentiment') {
        y = rows.map(row => row.roberta_relative_sentiment);
        title = 'Sentiment z';
        note = 'Relative to the saved episode reference population. Zero is its average, not neutral sentiment.';
    } else if (metric === 'comments') {
        y = rows.map(row => row.num_comments);
        title = 'Comments';
        note = 'Distinct comments mentioning this islander, including discussion after their exit.';
    } else if (metric === 'kalshi') {
        // --- ADDED KALSHI ---
        y = rows.map(row => row.kalshi_price);
        title = 'Kalshi price (¢)';
        note = 'Implied probability (in cents) of winning the finale, according to prediction markets.';
    } else {
        const forecasts = data.probabilities.filter(row => row.target_islander === selectedName).sort((a, b) => a.target_episode - b.target_episode);
        x = forecasts.map(row => row.target_episode);
        y = forecasts.map(row => row.booted_next_episode_probability == null ? null : row.booted_next_episode_probability * 100);
        title = 'Exit probability %';
        note = 'X-axis is the predicted episode. Predictions use the previous episode; gaps are unavailable, not zero.';
    }
    
    const layout = smallLayout(title);
    const episode = episodeNumber();
    if (episode != null) {
        const marker = metric === 'exit' ? episode + 1 : episode;
        layout.shapes = [{type: 'line', x0: marker, x1: marker, y0: 0, y1: 1, yref: 'paper', line: {color: '#d18a66', dash: 'dot', width: 1}}];
    }
    
    Plotly.react('trend-chart', [{x, y, type: 'scatter', mode: 'lines+markers', connectgaps: false, line: {color: '#287268', width: 2}, marker: {size: 4}, hovertemplate: 'Episode %{x}<br>%{y:.2f}<extra></extra>'}], layout, {responsive: true, displayModeBar: false});
    byId('trend-note').textContent = note;
}

function renderModelCards() {
    const episode = episodeNumber();
    const forecast = data.probabilities.find(row => row.target_islander === selectedName && row.feature_episode === episode);
    const survival = data.cox.find(row => row.target_islander === selectedName && row.calendar_episode === episode && row.within_observed_followup);
    let forecastNote = 'Select an episode to see the following episode’s estimate.';
    
    if (episode != null) {
        if (forecast?.prediction_status === 'predicted') {
            forecastNote = `Predicts final exit in episode ${episode + 1}, using episode ${episode}. Training outcomes through episode ${forecast.training_last_outcome_episode}.`;
        } else if (forecast) {
            forecastNote = 'Unavailable: not enough earlier exits to fit the historical model.';
        } else {
            forecastNote = 'Unavailable: no saved next-episode forecast for this view (outside observed follow-up or season end).';
        }
    }
    
    byId('model-cards').innerHTML = `
        <div class="model-card">
            <h3>Logistic · next-episode final exit</h3>
            <strong>${percent(forecast?.booted_next_episode_probability)}</strong>
            <p>${forecastNote}</p>
            ${forecast?.survive_next_episode_probability != null ? `<p>Survive next episode: ${percent(forecast.survive_next_episode_probability)}</p>` : ''}
        </div>
        <div class="model-card">
            <h3>Cox · survival through selected episode</h3>
            <strong>${percent(survival?.survival_probability)}</strong>
            <p>${episode == null ? 'Select an episode for a point estimate.' : 'Conditional survival from the baseline landmark; full-season descriptive fit, not next-episode risk.'}</p>
        </div>`;
    
    const evaluation = data.evaluation.find(row => row.model === 'logistic');
    const baseline = data.evaluation.find(row => row.model === 'training_rate_baseline');
    
    byId('model-quality').textContent = `Historical test: ${evaluation.test_rows} rows, ${evaluation.observed_exits} exits. Brier: ${evaluation.brier_score.toFixed(4)} vs baseline ${baseline.brier_score.toFixed(4)} (lower is better). AUC: ${evaluation.roc_auc.toFixed(3)}. 34 contestants; correlated outcomes; no external-season validation.`;
    
    if (byId('models').open) renderSurvival();
}

function renderSurvival() {
    const rows = data.cox.filter(row => row.target_islander === selectedName && row.within_observed_followup).sort((a, b) => a.tenure_episode - b.tenure_episode);
    const layout = smallLayout('Survival probability');
    
    layout.xaxis.title.text = 'Episodes since entrance (tenure)';
    layout.yaxis.range = [0, 1.03];
    layout.showlegend = true;
    layout.legend = {orientation: 'h', x: 0, y: 1.22, font: {size: 10}};
    layout.margin.t = 32;
    
    const km = [...data.km].sort((a, b) => a.tenure_episode - b.tenure_episode);
    
    Plotly.react('survival-chart', [
        {x: rows.map(row => row.tenure_episode), y: rows.map(row => row.survival_probability), mode: 'lines', name: 'Cox · selected islander', line: {color: '#287268', shape: 'hv'}},
        {x: km.map(row => row.tenure_episode), y: km.map(row => row.survival_probability), mode: 'lines', name: 'KM · whole cast', line: {color: '#be8c65', dash: 'dot', shape: 'hv'}}
    ], layout, {responsive: true, displayModeBar: false});
}

// Keep Plotly redraws in order when the arrows are clicked rapidly.
// Pending intermediate views can be skipped; the most recent view wins.
let renderQueue = Promise.resolve();
let latestRenderRequest = 0;
function renderAll(reset = false) {
    const requestNumber = ++latestRenderRequest;
    renderQueue = renderQueue.then(async () => {
        if (requestNumber !== latestRenderRequest) return;
        await renderNetwork(reset);
        if (requestNumber !== latestRenderRequest) return;
        renderSidebar();
    });
    return renderQueue;
}

// Event Listeners
byId('islander').addEventListener('change', () => {
    selectedName = byId('islander').value;
    selectedEdge = null;
    renderAll(byId('focus').checked);
});

byId('density').addEventListener('change', () => renderAll());

byId('departed').addEventListener('change', () => {
    selectedEdge = null;
    renderAll(true);
});

byId('focus').addEventListener('change', () => renderAll(true));

byId('reset').addEventListener('click', () => {
    byId('focus').checked = false;
    renderAll(true);
});

byId('trend').addEventListener('change', renderTrend);

byId('models').addEventListener('toggle', () => {
    if (byId('models').open) renderSurvival();
});

// Initial Render
updateViewText();
renderAll().then(() => {
    byId('build-notice').hidden = true;
});
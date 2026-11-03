/* Playback applies backpressure: at most one requested frame awaits a response. */
(function (root) {
    "use strict";
    let rendered = null;
    function acknowledge(frame) {
        rendered = frame;
        return frame;
    }
    function transport(scene, ticks, clicks, reset, experiment, duration, method,
                       frame, playing, counts) {
        const dc = root.dash_clientside;
        const unchanged = () => Array(6).fill(dc.no_update);
        const changes = (dc.callback_context.triggered || []).map(t => t.prop_id.split(".")[0]).filter(Boolean);
        const count = (counts || {})[scene];
        if (!count) return unchanged();
        if (!changes.length || changes.includes("scene") || changes.includes("reset")) {
            const marks = {};
            for (let i = 0; i < count; i += 10) marks[i] = String(i);
            return [0, count - 1, marks, true, "▶ Play", false];
        }
        // Selection changes win over a simultaneous timer tick or Play click.
        if (changes.some(id => ["experiment", "duration", "method"].includes(id))) {
            return [dc.no_update, dc.no_update, dc.no_update, true, "▶ Play", false];
        }
        if (changes.includes("play")) {
            playing = !playing;
            return [dc.no_update, dc.no_update, dc.no_update, !playing,
                    playing ? "Ⅱ Pause" : "▶ Play", playing];
        }
        if (changes.includes("clock") && playing && rendered &&
            rendered.scene === scene && rendered.frame === Number(frame) &&
            rendered.experiment === experiment && rendered.duration === Number(duration) &&
            rendered.method === method) {
            return [(Number(frame) + 1) % count, dc.no_update, dc.no_update,
                    dc.no_update, dc.no_update, dc.no_update];
        }
        return unchanged();
    }
    if (typeof module !== "undefined" && module.exports) module.exports = {transport, acknowledge};
    root.dash_clientside = Object.assign({}, root.dash_clientside, {
        replay: Object.assign({}, (root.dash_clientside || {}).replay, {transport, acknowledge})
    });
})(typeof window !== "undefined" ? window : globalThis);

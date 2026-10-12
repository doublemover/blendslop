/** Display-only pixel-cell contours. Never used to calculate silhouette scores. */
export function maskContours(mask, width, height) {
    if (mask.length !== width * height) throw new Error('Mask dimensions differ');
    const edges = [], outgoing = new Map();
    const key = (x, y) => y * (width + 1) + x;
    function add(x, y, nx, ny, direction) {
        const edge = {x, y, nx, ny, direction, used: false};
        edges.push(edge);
        const start = key(x, y);
        if (!outgoing.has(start)) outgoing.set(start, []);
        outgoing.get(start).push(edge);
    }
    const filled = (x, y) => x >= 0 && y >= 0 && x < width && y < height && mask[y * width + x];
    for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
        if (!filled(x, y)) continue;
        if (!filled(x, y - 1)) add(x, y, x + 1, y, 0);
        if (!filled(x + 1, y)) add(x + 1, y, x + 1, y + 1, 1);
        if (!filled(x, y + 1)) add(x + 1, y + 1, x, y + 1, 2);
        if (!filled(x - 1, y)) add(x, y + 1, x, y, 3);
    }
    const contours = [];
    for (const start of edges) {
        if (start.used) continue;
        const points = [[start.x, start.y]];
        let edge = start;
        while (edge) {
            edge.used = true;
            points.push([edge.nx, edge.ny]);
            if (edge.nx === start.x && edge.ny === start.y) break;
            const candidates = (outgoing.get(key(edge.nx, edge.ny)) || []).filter(next => !next.used);
            // Keep diagonally touching foreground cells separate: prefer a right turn.
            const rank = next => [1, 0, 3, 2].indexOf((next.direction - edge.direction + 4) % 4);
            candidates.sort((a, b) => rank(a) - rank(b));
            edge = candidates[0];
        }
        contours.push(points);
    }
    return contours;
}

export function strokeContours(context, contours, {color, alpha = 1, dash = []}) {
    context.save();
    context.strokeStyle = color;
    context.globalAlpha = alpha;
    context.lineWidth = 1.5;
    context.lineJoin = 'round';
    context.setLineDash(dash);
    for (const points of contours) {
        context.beginPath();
        context.moveTo(...points[0]);
        for (const point of points.slice(1)) context.lineTo(...point);
        context.stroke();
    }
    context.restore();
}

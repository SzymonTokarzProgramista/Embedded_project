async function capture() {
    const resp = await fetch('/camera/snapshot.jpg');
    const blob = await resp.blob();
    const url = URL.createObjectURL(blob);
    document.getElementById('photo').src = url;
}

async function save() {
    const resp = await fetch('/camera/save', { method: 'POST' });
    const text = await resp.text();
    alert(text);
}

document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('capture-btn').addEventListener('click', capture);
    document.getElementById('save-btn').addEventListener('click', save);
});

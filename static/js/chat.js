function formatBotMessage(response) {
    let html = '<div class="message bot">';
    
    if (response.content && response.content.sections) {
        response.content.sections.forEach(section => {
            html += `<div class="section">`;
            if (section.title) {
                html += `<div class="title">${section.title}</div>`;
            }
            if (section.points && section.points.length > 0) {
                html += '<ul>';
                section.points.forEach(point => {
                    html += `<li>${point}</li>`;
                });
                html += '</ul>';
            }
            html += '</div>';
        });
    } else {
        html += `<div class="section"><div class="title">ОШИБКА</div><ul><li>${response}</li></ul></div>`;
    }
    
    html += '</div>';
    return html;
}

function formatUserMessage(message) {
    return `<div class="message user">${message}</div>`;
} 
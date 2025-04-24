const intervals = [0, 1, 2, 3, 4, 5, Infinity]
const pos = [[27.5, 40], [40, 10], [55, 4], [70, 2], [85, 4], [100, 10], [112.5, 40]]
const fps = 100;
const delay = 1000;



const deltatime = delay / fps;
const height = window.innerHeight
const width = window.innerWidth

function moveObject(obj, start, end, transtime) {
    if (start == end) return;
    if ((start - end) != -1 && start - end != 1) {
        var startx = pos[start][0];
        var starty = pos[start][1];
        if (start > end) {
            var endx = pos[start + 1][0];
            var endy = pos[start + 1][1];
        }
        else{
            var endx = pos[start - 1][0];
            var endy = pos[start - 1][1];
        }
        var time = 0;
        var fulltime = delay / 2;
        let timer1 = setInterval(function() {
            time += deltatime;
            var curx = startx - (startx - endx) * time / fulltime;
            var cury = starty - (starty - endy) * time / fulltime;
            obj.style.right = cury + 'vh';
            obj.style.top = curx + 'vh';
            if (time >= fulltime) {
                clearInterval(timer1);
                if (start > end) {
                    startx = pos[end - 1][0];
                    starty = pos[end - 1][1];
                }
                else{
                    startx = pos[end + 1][0];
                    starty = pos[end + 1][1];
                }
                endx = pos[end][0];
                endy = pos[end][1];
                time = 0;
                fulltime = delay / 2;
                let timer2 = setInterval(function() {
                    time += deltatime;
                    var curx = startx - (startx - endx) * time / fulltime;
                    var cury = starty - (starty - endy) * time / fulltime;
                    obj.style.right = cury + 'vh';
                    obj.style.top = curx + 'vh';
                    if (time >= fulltime) clearInterval(timer2);
                }, deltatime);
                return;
            }
        }, deltatime);
        return;
    }
    var startx = pos[start][0];
    var starty = pos[start][1];
    var endx = pos[end][0];
    var endy = pos[end][1];
    var time = 0;
    var fulltime = delay;
    let timer = setInterval(function() {
        time += deltatime;
        var curx = startx - (startx - endx) * time / fulltime;
        var cury = starty - (starty - endy) * time / fulltime;
        obj.style.right = cury + 'vh';
        obj.style.top = curx + 'vh';
        if (time >= fulltime) clearInterval(timer);
    }, deltatime);
}
var curpos = 0;
function change_pos(position) {
    var startposition = [0, 0, 0, 0, 0];
    for(var i = 0; i < 5; i++) {
        startposition[i] = (i + curpos + 5) % 5 + 1;
    }
    var endposition = [0, 0, 0, 0, 0];
    for(var i = 0; i < 5; i++) {
        endposition[i] = (i + position + 5) % 5 + 1;
    }
    console.log(startposition, endposition);
    for (var i = 1; i <= 5; i++) {
        moveObject(document.getElementById("label" + i), startposition[i - 1], endposition[i - 1])
    }
    curpos = position
    setTimeout(function(){
        for (var i = 1; i <= 5; i++) {
            document.getElementById("label" + i).style.fontWeight = 300;
        }
        var index = 0
        while (endposition[index] != 3) index++
        index++
        document.getElementById("label" + index).style.fontWeight = 600;
    }, delay)
    
}
function getinterval(value){
    var index = 0;
    while (intervals[index] <= value) index++;
    return index;
}
var lastinterval = 0;
setInterval(function() { 
    var scroll = window.pageYOffset;
    var value = scroll / height;
    var interval = getinterval(value);
    if (interval == lastinterval) return;
    if (interval - lastinterval > 1) {
        lastinterval = lastinterval + 1
    }
    else if (lastinterval - interval > 1) lastinterval = lastinterval - 1;
    else lastinterval = interval;
    change_pos(lastinterval - 1);
}, delay);

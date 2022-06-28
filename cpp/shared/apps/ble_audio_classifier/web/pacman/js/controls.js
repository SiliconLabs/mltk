

function simulateKeyup(code) { 
    var e = jQuery.Event("keyup");
    e.keyCode = code;
    jQuery('body').trigger(e);
}
function simulateKeydown(code) { 
    var e = jQuery.Event("keydown");
    e.keyCode = code;
    jQuery('body').trigger(e);
}

function simulateKeyLeft() {
    simulateKeydown(37);
    simulateKeyup(13);
}

function simulateArrowRight() {
    simulateKeydown(39);
    simulateKeyup(13);
}

function simulateArrowUp() {
    simulateKeydown(38);
    simulateKeyup(13);
}

function simulateArrowDown() {
    simulateKeydown(40);
    simulateKeyup(13);
}

function simulatePKey() {
    simulateKeydown(80);
    simulateKeyup(13);
}

function simulateEnterKey() {
    simulateKeydown(13);
    simulateKeyup(13);
}

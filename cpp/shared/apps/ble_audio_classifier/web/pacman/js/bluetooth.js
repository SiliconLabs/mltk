const serviceUuid = "c20ffe90-4ed4-46b9-8f6c-ec143fce3e4e";
let _ble;
let _bleIsConnecting = false;
let _bleIsConnected = false;


function bluetoothInit() {
    _ble = new p5ble();
}

function bluetoothIsReady() {
    return _bleIsConnected && !_bleIsConnecting;
}

function bluetoothTryConnect() {
    if(_bleIsConnecting) {
        return;
    }

    if(HOME) {
        playEatPillSound();
    }

    _bleIsConnecting = true;
    console.log('Attempting to connect to BLE device ...');
    // Create a p5ble class
    

    // Connect to a device by passing the service UUID
    _ble.connect(serviceUuid, gotCharacteristics);

    
    $("#bluetooth-msg").text("Connecting to Bluetooth ...");
}

function onDisconnected() {
    console.log('Device got disconnected.');
    $("#bluetooth-msg").text("Bluetooth disconnected, click to reconnect");
    $("#bluetooth-msg").fadeIn('slow');
    $("#command-log-dialog").fadeOut('slow');
    _bleIsConnected = false;
    if (!HOME) { 
        pauseGame();
    }
}

  // A function that will be called once got characteristics
function gotCharacteristics(error, characteristics) {
    _bleIsConnecting = false;
    if (error || !characteristics) {
        console.log('error: ', error);
        $("#bluetooth-msg").text("Bluetooth failed, click to try again");
        return;
    }
     
    console.log('characteristics: ', characteristics);
  
    // Check if myBLE is connected
    if(_ble.isConnected()) {
        _bleIsConnected = true;

        $("#command-log").val('');
        $("#command-log-dialog").fadeIn('slow');
       
        if(HOME) {
            $("#bluetooth-msg").text("Say \"go\" to begin");
            playReadySound();
        } else {
            $("#bluetooth-msg").fadeOut('slow');
            resumeGame();
        }
        
        let myCharacteristic = characteristics[0];
        _ble.startNotifications(myCharacteristic, handleNotifications, 'string');
    }

    // Add a event handler when the device is disconnected
    _ble.onDisconnected(onDisconnected)
}

function handleNotifications(data) {
    console.log('data: ', data);
    let toks = data.split(',');
    let command = parseInt(toks[0]);
    let confidence = parseInt(toks[1]);
    let commandName;

    switch(command) {
        case 0:
            commandName = 'Left';
            simulateKeyLeft();
            break; 
        case 1: 
            commandName = 'Right';
            simulateArrowRight();
            break; 
        case 2:
            commandName = 'Up';
            simulateArrowUp();
            break; 
        case 3:
            commandName = 'Down';
            simulateArrowDown();
            break; 
        case 4:
            commandName = 'Stop';
            simulatePKey();
            break; 
        case 5:
            commandName = 'Go';
            simulateEnterKey();
            break; 
        default:
            break
    }

    var box = $("#command-log");
    let confidencePercentage = (confidence / 255.)  * 100;
    let txt = `${commandName.padEnd(15, ' ')} ${confidencePercentage.toFixed(1)}%\n`;
    box.val(txt + box.val());
  }
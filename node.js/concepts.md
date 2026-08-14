# Asynchronous Javascript
Possibly long-running executions:
- http request through ~fetch()~
- user camera or microphone access ~getUserMedia()~
- file selection through ~showOpenFilePicker()~

## Callbacks
Functions that will be run within the execution of another, similar to an event handler. [callbacks.js](./callbacks.js)

_Prone to callback hell but still necessary in some performance related scenarios:_

> [!TIP]
> modules like fs, crypto or zlib rely heavily in the error-first pattern of the callback, avoiding minor memory overhead

## Promises
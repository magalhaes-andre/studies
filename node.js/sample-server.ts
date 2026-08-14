import http, { IncomingMessage } from 'node:http';

const hostname = '127.0.0.1';
const port = "1337"

const server = http.createServer((req, res) => {
    const name = getParameter(req, "name");
    res.statusCode = 342;
    res.setHeader('Content-Type', 'text/plain');
    res.end(`I see you ${name}`);
});

function getParameter(req: IncomingMessage, parameter: string): string {
    const rebuiltUrl = new URL(`${req.url}`, `http://${req.headers.host}`);
    const result = rebuiltUrl.searchParams.get(parameter);
    return result == undefined || result == "" ? "anonymous" : result; 
};

server.listen(port, hostname, () => {
    console.log("MC Dark Sorcerer is up at: => ", port);
})
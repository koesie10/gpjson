const fs = require('fs');
const path = require('path');

const filename = path.resolve(__dirname, '../datasets/twitter_small_records_smaller.json');
const query = '$.user.lang';
const warmupRuns = 5;
const actualRuns = 10;

const filesize = fs.statSync(filename).size;

// https://stackoverflow.com/a/18650828
function formatBytes(bytes, decimals = 2) {
    if (bytes === 0) return '0 bytes';

    const k = 1000;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ['bytes', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB'];

    const i = Math.floor(Math.log(bytes) / Math.log(k));

    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
}

const gpjson = Polyglot.eval('gpjson', 'jsonpath');

// Warmup
for (let i = 0; i < warmupRuns; i++) {
    gpjson.querySequential(filename, query);
}

console.log('Warm up done');

const start = process.hrtime.bigint();

for (let i = 0; i < actualRuns; i++) {
    gpjson.querySequential(filename, query);
}

const end = process.hrtime.bigint();

const totalTimeMilliseconds = Number(end - start) / 1_000_000;

const averageTimeMilliseconds = totalTimeMilliseconds / actualRuns;
const averageTimeSeconds = averageTimeMilliseconds / 1_000;
const speed = filesize / averageTimeSeconds;

console.log(`Average time is ${averageTimeMilliseconds.toFixed(3).toString()}ms, ${formatBytes(speed)}/second`);

fs.writeFileSync('./benchmark_profile.json', gpjson.exportTimings());

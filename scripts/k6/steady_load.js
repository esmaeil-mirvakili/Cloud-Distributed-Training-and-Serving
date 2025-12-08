import http from 'k6/http';
import { sleep, check } from 'k6';

const BASE_URL = __ENV.SERVICE_URL || 'http://smirvaki-llama-serving:80';
const TIMEOUT = __ENV.TIMEOUT || '120s';
const PROMPT = __ENV.PROMPT || 'Summarize the premise of the book Dune in one sentence.';

export const options = {
  vus: Number(__ENV.VUS || 50),
  duration: __ENV.DURATION || '10m',
  thresholds: {
    http_req_failed: ['rate<0.02'],
  },
};

const payload = JSON.stringify({
  model: 'llama',
  messages: [{ role: 'user', content: PROMPT }],
  temperature: 0.1,
  max_tokens: 128,
});
const params = {
  headers: { 'Content-Type': 'application/json' },
  timeout: TIMEOUT,
};

export default function () {
  const res = http.post(`${BASE_URL}/v1/chat/completions`, payload, params);
  check(res, { 'status is 200': (r) => r.status === 200 });
  sleep(1);
}

import http from 'k6/http';
import { sleep, check } from 'k6';

const BASE_URL = __ENV.SERVICE_URL || 'http://smirvaki-llama-serving:80';
const TIMEOUT = __ENV.TIMEOUT || '120s';
const PROMPT = __ENV.PROMPT || 'Write a two-sentence LinkedIn post about cloud cost optimization.';

export const options = {
  stages: [
    { duration: '30s', target: Number(__ENV.BASE_VUS || 20) },
    { duration: '1m', target: Number(__ENV.SPIKE_VUS || 150) },
    { duration: '180s', target: Number(__ENV.SPIKE_VUS || 150) },
    { duration: '1m', target: Number(__ENV.BASE_VUS || 20) },
  ],
  thresholds: {
    http_req_failed: ['rate<0.05'],
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

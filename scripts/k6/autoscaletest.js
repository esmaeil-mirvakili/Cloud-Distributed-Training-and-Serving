import http from 'k6/http';
import { sleep, check } from 'k6';

export const options = {
  vus: 10,
  stages: [
    { duration: '1m', target: 20 },
    { duration: '3m', target: 20 },
    { duration: '1m', target: 0 },
  ],
  thresholds: {
    http_req_duration: ['p(95)<1500'],
    http_req_failed: ['rate<0.01'],
  },
};

const BASE_URL = __ENV.SERVICE_URL || 'http://smirvaki-llama-serving:80';
const PAYLOAD = JSON.stringify({
  prompt: 'Say hello in one sentence.',
  n_predict: 64,
  temperature: 0.7,
});
const params = {
  headers: { 'Content-Type': 'application/json' },
  timeout: '1000s',
};

export default function () {
  const res = http.post(`${BASE_URL}/completion`, PAYLOAD, params);
  check(res, { 'status is 200': (r) => r.status === 200 });
  sleep(1);
}

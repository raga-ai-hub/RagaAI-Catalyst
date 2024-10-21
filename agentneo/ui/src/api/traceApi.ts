import axios from 'axios';
import initSqlJs from 'sql.js';
import { TimelineData } from '../types/timeline';
import { initDatabase } from '../utils/databaseUtils';

const API_BASE_URL = '/api'; // This will be proxied to our backend server

export const fetchProjects = async () => {
  const response = await axios.get(`${API_BASE_URL}/projects`);
  return response.data;
};

export const fetchTraces = async (projectId: number) => {
  const response = await axios.get(`${API_BASE_URL}/traces`, { params: { projectId } });
  return response.data;
};

export const fetchTraceDetails = async (traceId: number) => {
  const response = await axios.get(`${API_BASE_URL}/traces/${traceId}`);
  return response.data;
};

export const fetchTimelineData = async (traceId: string) => {
  const response = await axios.get(`${API_BASE_URL}/timeline/${traceId}`);
  return response.data;
};

export const fetchLLMCalls = async (traceId: number) => {
  const response = await axios.get(`${API_BASE_URL}/llm-calls`, { params: { traceId } });
  return response.data;
};

export const fetchToolCalls = async (traceId: number) => {
  const response = await axios.get(`${API_BASE_URL}/tool-calls`, { params: { traceId } });
  return response.data;
};

export const fetchAgentCalls = async (traceId: number) => {
  const response = await axios.get(`${API_BASE_URL}/agent-calls`, { params: { traceId } });
  return response.data;
};

export const fetchErrors = async (traceId: number) => {
  const response = await axios.get(`${API_BASE_URL}/errors`, { params: { traceId } });
  return response.data;
};


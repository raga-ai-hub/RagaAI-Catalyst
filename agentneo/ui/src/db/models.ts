const { DataTypes } = require('sequelize');

module.exports = (sequelize) => {
  const TraceModel = sequelize.define('Trace', {
    id: {
      type: DataTypes.INTEGER,
      primaryKey: true,
      autoIncrement: true
    },
    project_id: {
      type: DataTypes.INTEGER,
      allowNull: false
    },
    start_time: {
      type: DataTypes.DATE,
      defaultValue: DataTypes.NOW
    },
    end_time: {
      type: DataTypes.DATE,
      allowNull: true
    },
    duration: {
      type: DataTypes.FLOAT,
      allowNull: true
    }
  }, {
    tableName: 'traces'
  });

  const LLMCallModel = sequelize.define('LLMCall', {
    id: {
      type: DataTypes.INTEGER,
      primaryKey: true,
      autoIncrement: true
    },
    project_id: {
      type: DataTypes.INTEGER,
      allowNull: false
    },
    agent_id: {
      type: DataTypes.INTEGER,
      allowNull: true
    },
    name: {
      type: DataTypes.STRING,
      allowNull: false
    },
    model: {
      type: DataTypes.STRING,
      allowNull: true
    },
    input_prompt: {
      type: DataTypes.TEXT,
      allowNull: false
    },
    output: {
      type: DataTypes.TEXT,
      allowNull: false
    },
    tool_call: {
      type: DataTypes.STRING,
      allowNull: true
    },
    start_time: {
      type: DataTypes.DATE,
      defaultValue: DataTypes.NOW
    },
    end_time: {
      type: DataTypes.DATE,
      allowNull: true
    },
    duration: {
      type: DataTypes.FLOAT,
      allowNull: true
    },
    token_usage: {
      type: DataTypes.JSON,
      allowNull: false
    },
    cost: {
      type: DataTypes.JSON,
      allowNull: false
    },
    memory_used: {
      type: DataTypes.INTEGER,
      allowNull: false
    },
    trace_id: {
      type: DataTypes.INTEGER,
      allowNull: false
    }
  }, {
    tableName: 'llm_call'
  });

  const ToolCallModel = sequelize.define('ToolCall', {
    id: {
      type: DataTypes.INTEGER,
      primaryKey: true,
      autoIncrement: true
    },
    project_id: {
      type: DataTypes.INTEGER,
      allowNull: false
    },
    agent_id: {
      type: DataTypes.INTEGER,
      allowNull: true
    },
    name: {
      type: DataTypes.STRING,
      allowNull: false
    },
    input_parameters: {
      type: DataTypes.TEXT,
      allowNull: false
    },
    output: {
      type: DataTypes.TEXT,
      allowNull: false
    },
    start_time: {
      type: DataTypes.DATE,
      defaultValue: DataTypes.NOW
    },
    end_time: {
      type: DataTypes.DATE,
      allowNull: true
    },
    duration: {
      type: DataTypes.FLOAT,
      allowNull: true
    },
    memory_used: {
      type: DataTypes.INTEGER,
      allowNull: false
    },
    trace_id: {
      type: DataTypes.INTEGER,
      allowNull: false
    },
    network_calls: {
      type: DataTypes.JSON,
      allowNull: true
    }
  }, {
    tableName: 'tool_call'
  });

  const AgentCallModel = sequelize.define('AgentCall', {
    id: {
      type: DataTypes.INTEGER,
      primaryKey: true,
      autoIncrement: true,
      unique: true
    },
    project_id: {
      type: DataTypes.INTEGER,
      allowNull: false
    },
    name: {
      type: DataTypes.STRING,
      allowNull: false
    },
    start_time: {
      type: DataTypes.DATE,
      defaultValue: DataTypes.NOW
    },
    end_time: {
      type: DataTypes.DATE,
      allowNull: true
    },
    trace_id: {
      type: DataTypes.INTEGER,
      allowNull: false
    },
    llm_call_ids: {
      type: DataTypes.JSON,
      allowNull: true
    },
    tool_call_ids: {
      type: DataTypes.JSON,
      allowNull: true
    },
    user_interaction_ids: {
      type: DataTypes.JSON,
      allowNull: true
    }
  }, {
    tableName: 'agent_call'
  });

  const ErrorModel = sequelize.define('Error', {
    id: {
      type: DataTypes.INTEGER,
      primaryKey: true,
      autoIncrement: true
    },
    project_id: {
      type: DataTypes.INTEGER,
      allowNull: false
    },
    trace_id: {
      type: DataTypes.INTEGER,
      allowNull: false
    },
    agent_id: {
      type: DataTypes.INTEGER,
      allowNull: true
    },
    tool_call_id: {
      type: DataTypes.INTEGER,
      allowNull: true
    },
    llm_call_id: {
      type: DataTypes.INTEGER,
      allowNull: true
    },
    error_type: {
      type: DataTypes.STRING,
      allowNull: false
    },
    error_message: {
      type: DataTypes.TEXT,
      allowNull: false
    },
    timestamp: {
      type: DataTypes.DATE,
      defaultValue: DataTypes.NOW
    }
  }, {
    tableName: 'errors'
  });

  // Define associations
  TraceModel.hasMany(LLMCallModel, { as: 'llm_calls', foreignKey: 'trace_id' });
  TraceModel.hasMany(ToolCallModel, { as: 'tool_calls', foreignKey: 'trace_id' });
  TraceModel.hasMany(AgentCallModel, { as: 'agent_calls', foreignKey: 'trace_id' });
  TraceModel.hasMany(ErrorModel, { as: 'errors', foreignKey: 'trace_id' });

  LLMCallModel.belongsTo(TraceModel, { foreignKey: 'trace_id' });
  ToolCallModel.belongsTo(TraceModel, { foreignKey: 'trace_id' });
  AgentCallModel.belongsTo(TraceModel, { foreignKey: 'trace_id' });
  ErrorModel.belongsTo(TraceModel, { foreignKey: 'trace_id' });

  return {
    TraceModel,
    LLMCallModel,
    ToolCallModel,
    AgentCallModel,
    ErrorModel
  };
};
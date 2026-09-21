typedef struct {
    float distance_ratio;
} MotorParameters;

typedef struct {
    MotorParameters parameters;
} MotorModel;

typedef struct {
    float radians;
    float sin;
    float cos;
    float tan;
} SensorAngle;

typedef struct {
    float volts;
} Voltage;

typedef struct {
    float voltage_scale;
    Voltage voltage_offset;
} SensorParameters;

typedef struct {
    SensorParameters parameters;
} SensorModel;

typedef struct {
    float radians;
    float sin;
    float cos;
    float tan;
} ShaftAngle;

typedef struct {
    ShaftAngle shaft_angle_offset;
} MountedSensorParameters;

typedef struct {
    MountedSensorParameters mounted_alpha_sensor_parameters;
    MountedSensorParameters mounted_beta_sensor_parameters;
} AssemblyParameters;

typedef struct {
    MotorModel motor;
    SensorModel alpha_sensor;
    SensorModel beta_sensor;
    AssemblyParameters parameters;
} AssemblyModel;
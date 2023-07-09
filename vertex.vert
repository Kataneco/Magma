#version 460

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec3 color;

layout(location = 0) out vec4 fragmentColor;

layout(push_constant) uniform block {
    mat4 renderMatrix;
} constants;

void main() {
    gl_Position = constants.renderMatrix * vec4(position, 1.0f);
    fragmentColor = vec4(sin(constants.renderMatrix[3].x)+sin(position.x), sin(constants.renderMatrix[3].y)+sin(position.y), sin(constants.renderMatrix[3].z)+sin(position.z), 0.0f);
}

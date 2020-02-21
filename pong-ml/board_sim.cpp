
#include "pch.h"
#include "board_sim.h"

#include "config.h"

namespace pingpong
{
    namespace
    {
        vec2 to_vec2(cpVect v) { return {v.x, v.y}; }
    }

    BoardSim::BoardSim()
    {
        _space = cpSpaceNew();
        //cpSpaceSetGravity(_space, cpVect{0.1f, -0.01f});

        cpBody* staticBody = cpSpaceGetStaticBody(_space);

        // Create the left and right board boundaries.
        //
        cpVect wallVertices[] = {
            { -0.5, -1.0 },
            { +0.5, -1.0 },
            { +0.5, +1.0 },
            { -0.5, +1.0 } };

        cpShape* wallShapes[] = {
            cpSpaceAddShape(_space, cpPolyShapeNew(staticBody, 4, wallVertices, cpTransform{1.0, 0.0, 0.0, 1.0, -1.0, 0.0}, 0.0)),
            cpSpaceAddShape(_space, cpPolyShapeNew(staticBody, 4, wallVertices, cpTransform{1.0, 0.0, 0.0, 1.0, +1.0, 0.0}, 0.0)) };
        for (auto* wallShape : wallShapes) {
            cpShapeSetElasticity(wallShape, cfg.wallsElasticity);
        }

        // Create the ball.
        //
        _ballBody = cpSpaceAddBody(_space, cpBodyNew(cfg.ballMass, cpMomentForCircle(cfg.ballMass, cfg.ballRadius, cfg.ballRadius, cpvzero)));
        cpShape* ballShape = cpSpaceAddShape(_space, cpCircleShapeNew(_ballBody, cfg.ballRadius, cpvzero));
        cpShapeSetElasticity(ballShape, cfg.ballElasticity);

        // Create the racquets.
        //
        for (int playerSlot = 0; playerSlot < 2; ++playerSlot)
        {
            auto& racquet = _racquets[playerSlot];
            const auto sideFactor = 2.0 * static_cast<double>(playerSlot) - 1.0;

            racquet.gripBody = cpSpaceAddBody(_space, cpBodyNew(cfg.racquetGripMass, cpMomentForCircle(cfg.racquetGripMass, cfg.racquetGripRadius, cfg.racquetGripRadius, cpvzero)));
            cpShape* racquetGripShape = cpSpaceAddShape(_space, cpCircleShapeNew(racquet.gripBody, cfg.racquetGripRadius, cpvzero));
            cpShapeSetElasticity(racquetGripShape, cfg.racquetGripElasticity);

            cpVect racquetFaceVertices[] = {
                { -0.5 * sideFactor * cfg.racquetFaceDim.x, -0.5 * sideFactor * cfg.racquetFaceDim.y },
                { +0.5 * sideFactor * cfg.racquetFaceDim.x, -0.5 * sideFactor * cfg.racquetFaceDim.y },
                { +0.5 * sideFactor * cfg.racquetFaceDim.x, +0.5 * sideFactor * cfg.racquetFaceDim.y },
                { -0.5 * sideFactor * cfg.racquetFaceDim.x, +0.5 * sideFactor * cfg.racquetFaceDim.y } };

            racquet.faceBody = cpSpaceAddBody(_space, cpBodyNew(cfg.racquetFaceMass, cpMomentForPoly(cfg.racquetFaceMass, 4, racquetFaceVertices, cpvzero, 0.0)));
            cpShape* racquetFaceShape = cpSpaceAddShape(_space, cpPolyShapeNew(racquet.faceBody, 4, racquetFaceVertices, cpTransformIdentity, 0.0));
            cpShapeSetElasticity(racquetFaceShape, cfg.racquetFaceElasticity);

            // The joint constraints are created later (in NewMatch() -> ResetSpace()), after proper object repositioning.
        }

        _state.match = -1;   // Shall be incremented to 0 in the next line.
    }

    BoardSim::~BoardSim()
    {
        if (_space != nullptr)
            cpSpaceDestroy(_space);
    }

    void BoardSim::NewMatch()
    {
        ++_state.match;
        _state.time = 0;
        _state.terminal = false;
        _state.winner = -1;

        ResetSpace();
        UpdateState();
    }

    void BoardSim::Step(std::array<uint8_t, 2> actions)
    {
        assert(_state.match >= 0 && "Call BoardSim::NewMatch() first");

        const auto ballPos = cpBodyGetPosition(_ballBody);
        const auto ballVel = cpBodyGetVelocity(_ballBody);
        const auto ballResultantForce = cpVect{-cfg.ballMotionResistance * ballVel.x, -cfg.ballMotionResistance * ballVel.y};
        cpBodyApplyForceAtWorldPoint(_ballBody, ballResultantForce, ballPos);

        for (int playerSlot = 0; playerSlot < 2; ++playerSlot)
        {
            auto& racquet = _racquets[playerSlot];
            const auto sideFactor = 2.0 * static_cast<double>(playerSlot) - 1.0;

            const auto action = actions[playerSlot];
            const auto move = ivec2{(action % 3) - 1, (action / 3) - 1};

            auto move_r = vec2{static_cast<float>(move.x), static_cast<float>(move.y)};
            if (move.x != 0 && move.y != 0)
                move_r /= glm::length(move_r);

            const auto racquetGripPos = cpBodyGetPosition(racquet.gripBody);
            //const cpVect racquetGripVel = cpBodyGetVelocity(racquet.gripBody);
            const auto racquetGripResultantForce = cpVect{-sideFactor * cfg.racquetGripSteeringForce * move_r.x, -sideFactor * cfg.racquetGripSteeringForce * move_r.y};
            cpBodyApplyForceAtWorldPoint(racquet.gripBody, racquetGripResultantForce, racquetGripPos);
            cpBodySetAngularVelocity(racquet.gripBody, 0.0);

            const auto racquetFacePos = cpBodyGetPosition(racquet.faceBody);
            const cpVect racquetFaceVel = cpBodyGetVelocity(racquet.faceBody);
            const auto racquetFaceResultantForce = cpVect{-cfg.racquetFaceMotionResistance * racquetFaceVel.x, -cfg.racquetFaceMotionResistance * racquetFaceVel.y};
            cpBodyApplyForceAtWorldPoint(racquet.faceBody, racquetFaceResultantForce, racquetFacePos);
        }

        cpSpaceStep(_space, cfg.timeStep);

        ++_state.time;
        UpdateState();
    }

    void BoardSim::UpdateState()
    {
        {
            cpVect pos = cpBodyGetPosition(_ballBody);
            _state.ball.pos = vec2{pos.x, pos.y};

            cpVect vel = cpBodyGetVelocity(_ballBody);
            _state.ball.vel = vec2{vel.x, vel.y};

            _state.ball.angularVel = cpBodyGetAngularVelocity(_ballBody);
        }

        for (int playerSlot = 0; playerSlot < 2; ++playerSlot)
        {
            _state.racquets[playerSlot].grip.pos = to_vec2(cpBodyGetPosition(_racquets[playerSlot].gripBody));
            _state.racquets[playerSlot].grip.vel = to_vec2(cpBodyGetVelocity(_racquets[playerSlot].gripBody));
            _state.racquets[playerSlot].face.pos = to_vec2(cpBodyGetPosition(_racquets[playerSlot].faceBody));
            _state.racquets[playerSlot].face.vel = to_vec2(cpBodyGetVelocity(_racquets[playerSlot].faceBody));
            _state.racquets[playerSlot].face.angle = static_cast<float>(cpBodyGetAngle(_racquets[playerSlot].faceBody));
            _state.racquets[playerSlot].face.angularVel = static_cast<float>(cpBodyGetAngularVelocity(_racquets[playerSlot].faceBody));
        }

        CheckTerminalCondition();
    }

    void BoardSim::CheckTerminalCondition()
    {
        _state.scores = {0.0f, 0.0f};
        _state.penalties = {0.0f, 0.0f};

        if (_state.ball.pos.y < -1.0f) {
            _state.terminal = true;
            _state.winner = 1;
            _state.scores[1] = 100.0f;
        }
        else if (_state.ball.pos.y > 1.0f) {
            _state.terminal = true;
            _state.winner = 0;
            _state.scores[0] = 100.0f;
        }
        else if (_state.racquets[0].grip.pos.y < -1.0f || _state.racquets[0].grip.pos.y > 1.0f) {
            _state.terminal = true;
            _state.winner = 1;
            _state.penalties[0] = 50.0f;
        }
        else if (_state.racquets[1].grip.pos.y < -1.0f ||_state.racquets[1].grip.pos.y > 1.0f) {
            _state.terminal = true;
            _state.winner = 0;
            _state.penalties[1] = 50.0f;
        }
        else if (_state.time > cfg.matchTickLimit) {
            _state.terminal = true;
        }
    }

    void BoardSim::ResetSpace()
    {
        cpBodySetPosition(_ballBody, cpvzero);
        cpBodySetVelocity(_ballBody, cpvzero);
        cpBodySetAngle(_ballBody, 0.0);
        cpBodySetAngularVelocity(_ballBody, 0.0);

        for (int playerSlot = 0; playerSlot < 2; ++playerSlot)
        {
            auto& racquet = _racquets[playerSlot];
            const auto sideFactor = 2.0 * static_cast<double>(playerSlot) - 1.0;

            cpBodySetPosition(racquet.gripBody, cpVect{0.0, sideFactor * cfg.racquetToCenterDistance});
            cpBodySetVelocity(racquet.gripBody, cpvzero);
            cpBodySetAngle(racquet.gripBody, 0.0);
            cpBodySetAngularVelocity(racquet.gripBody, 0.0);
            cpBodySetPosition(racquet.faceBody, cpVect{0.0, sideFactor * (0.5 * cfg.racquetFaceDim.y + cfg.racquetToCenterDistance + cfg.racquetGripRadius)});
            cpBodySetVelocity(racquet.faceBody, cpvzero);
            cpBodySetAngle(racquet.faceBody, 0.0);
            cpBodySetAngularVelocity(racquet.faceBody, 0.0);

            if (racquet.pivotJoint == nullptr) {
                racquet.pivotJoint = cpSpaceAddConstraint(_space, cpPivotJointNew2(racquet.gripBody, racquet.faceBody, cpVect{0.0, 0.0}, cpVect{0.0, sideFactor * (-0.5 * cfg.racquetFaceDim.y - cfg.racquetGripRadius)}));
            }
        }
    }

} // namespace pingpong

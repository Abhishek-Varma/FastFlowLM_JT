/// \file d3_engine_adapter.hpp
/// \brief The one place where the d3 contract's engine interface (decision/d3_engine.hpp) meets the real engine
///        (d3_npu, IRON FLM_DLL/detail/d3_npu).
/// \note  When the engine's multi-position output (`encode_state_once`, work/m2_api.md) lands, this file and
///        d3_engine_adapter.cpp are the only ones that change. Until then the adapter runs what the shipped
///        engine can: one hidden row per prompt (the last token) through `decision_model::encode_batch` with
///        `state_len`, which serves choice / noul / score blocks (with images in the state or in a block);
///        a Box block asks for four rows and is refused with a message that names the missing engine feature.
#pragma once

#include <memory>

#include "decision/d3_engine.hpp"
#include "decision_model.hpp"

/// \param engine the loaded d3 engine (not owned; must outlive the adapter)
std::unique_ptr<flm::d3::ReadoutEngine> make_d3_engine_adapter(decision_model* engine, int image_token_id_hint = 248056);

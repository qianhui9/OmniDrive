# 原始 VQA 语义依赖复核

固定种子按城市、词法类别和场景抽取 48 条，另对 3 条明确位置问题作定向文本复核。
以下是 AI 分析者逐条文本审阅，不是人工标注、图像指称确认或任务输入完备性认证。未审阅样本保留 Unknown。

| 原始问题 | 域 | 缺失依赖 |
|---|---|---|
| How does the position of the sun inform us about the time of day? | local | Sun position and time-of-day inference; no sun/lighting state |
| What does the overpass in the distance suggest about the road network? | global | Overpass appearance and road-network interpretation; no road topology |
| Is the vehicle in a position to turn soon, based on road markings or signage? | global | Turn feasibility requires markings, signs and routing; no legal-turn semantics |
| Describe the parking lot visible from your position. | regional | Parking-lot description includes appearance/layout; no parking-lot state |
| Can you describe the bus that is visible from the vehicle's position? | local | Bus description is open-ended appearance, not a direction answer |
| Describe the position and possible activity of the truck to the right-rear. | local | Truck position AND possible activity; geometry plus grounded motion/intent needed |
| Considering the layout of the road and the buildings, in what kind of city does this scene take place? | global | City type inferred from roads/buildings; no building or city semantics |
| What kind of road markings can you see, and what do they indicate? | regional | Marking recognition and legal meaning; lane control points lack both |
| Can you describe the barriers that are marking the lane restrictions? | regional | Barrier description and lane-restriction semantics; geometry insufficient |
| How does the road layout ahead appear to be? | regional | Open-ended forward road layout; topology/domain completeness Unknown |
| What type of road are we currently on? | regional | Road-type classification; no semantic road-type field |
| What kind of road signs or traffic signals can you see? | regional | Sign/signal discovery and semantics; no signs or light state |
| Based on the vehicles around us, can we infer the speed of traffic flow? | regional | Traffic-flow speed aggregates multiple vehicles; temporal motion and domain needed |
| Is the vehicle currently in motion, and if so, can you describe the driving behavior? | ego | Ego motion plus driving-behaviour description; behaviour semantics still missing |
| Is the traffic behind the vehicle dense or moving? | regional | Traffic density and motion behind ego; complete rear domain and time evidence |
| Based on the vehicles you can see, what is the likely speed limit in this area? | regional | Likely legal speed limit cannot be inferred from velocity predictions alone |
| What is the speed limit on the road you are on, if visible? | regional | Visible speed-limit sign OCR/legal limit absent from state |
| What kind of intersection is the vehicle approaching? | regional | Intersection type/topology absent from lane control points |
| What is the condition of traffic behind the vehicle? | regional | Rear traffic condition requires regional search and aggregate scene semantics |
| What is the condition of the road in front of the vehicle? | regional | Road condition includes surface/appearance beyond object geometry |
| What is the condition of the lot to the right of the vehicle? | regional | Lot condition is an open-ended appearance/domain question |
| Is the area behind the vehicle clear of traffic and pedestrians? | regional | Absence of traffic and pedestrians requires complete rear search domain |
| What is the traffic condition directly in front of you? | regional | Traffic condition ahead aggregates agents and may include signals |
| How does the traffic behind you compare to the traffic in front? | regional | Two-domain traffic comparison needs complete front and rear aggregates |
| What type of buildings can be seen from the vehicle's position? | regional | Building-type semantics absent from object/map state |
| What can be inferred about the location of this area? | global | Area-location inference is open-ended scene semantics |
| Is the vehicle in an appropriate position to proceed through the access control point? | global | Proceeding through access control requires gate/sign/rule/current hazard state |
| Is the vehicle currently in a position to safely change lanes if needed? | global | Safe lane change requires complete hazards, adjacent lanes, intent and gaps |
| What type of establishments can be seen from the vehicle's position? | regional | Establishment recognition absent from state |
| What establishments or buildings are visible from your position? | regional | Building/establishment search and semantics absent from state |
| What is the significance of the 'KEEP CLEAR' marking on the road? | local | KEEP CLEAR meaning is a road-marking/rule question, not map geometry |
| What is the condition of the road in terms of traffic? | regional | Traffic condition requires complete road-domain traffic evidence |
| What is the setting around the road? | regional | Road surroundings need appearance/building/land-use semantics |
| How does the road condition appear to be? | regional | Road condition is appearance-sensitive and open-ended |
| What is the condition of the road in the direction we're headed? | regional | Forward road condition needs surface/appearance evidence |
| What kind of buildings flank the road? | regional | Building recognition absent from the current decoded fields |
| What type of junction are you approaching? | regional | Junction type requires topology/semantic map labels |
| What's the function of the white gate we're approaching on the left? | local | White gate identity and function require colour/appearance/access-control semantics |
| Is the vehicle currently moving or stationary? | ego | Ego moving/stationary depends on ego velocity plus a defined decision boundary; M1 has no certified ego-motion fact |
| Is the vehicle currently moving or stationary? | ego | Ego moving/stationary depends on ego velocity plus a defined decision boundary; M1 has no certified ego-motion fact |
| What is the driver's responsibility when approaching the intersection or pedestrian crossing? | global | Driver responsibility is a policy/safety question requiring crossings and rules |
| Is the traffic moving in the lane to the right-rear of the vehicle? | regional | Right-rear lane traffic motion needs lane membership, multiple tracks and domain coverage |
| What does the road marking directly in front of the vehicle indicate? | local | Marking meaning absent from lane control points |
| Considering the empty road behind us, what could be the reason for this lack of traffic? | regional | Causal explanation of empty road is not a local direction fact |
| How is the road marked directly in front of you? | regional | Road-marking recognition absent from state |
| Can you describe the vehicle directly behind you? | local | Rear vehicle description needs unique visual referent and appearance attributes |
| What is the condition of the road behind you? | regional | Rear road condition requires regional appearance evidence |
| Describe the area to the right of the vehicle. | regional | Right-side area description requires scene/appearance semantics |

定向检查发现真实问题 “What is the position of the bus in relation to the vehicle?”。
这是有条件的单关系机会，仍未证明图像中只有一个可指称 bus，也未冻结自然语言“position”的完整答案要求。
黑色 SUV 位置需要外观语义；卡车位于哪条车道需要身份、车道几何与拓扑共同支持。
默认 Planning 与完整安全任务仍需要完整当前场景。当前 Head 提供预测字段，不提供完整语义、独立物理真值或查询指称。
不能用 48 个未完成视觉 grounding 的文本样本推断 4,881 条全部不可替代。已验证可省略语言输入为零，条件机会与未知部分单列。

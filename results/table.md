| Board | What was planted | Probes that fired |
| --- | --- | --- |
| Even board | Enough votes, spread evenly, no artefacts planted. | none |
| Too few votes | The same board on 200 votes instead of 6000. | `rank_volatility`, `separation` |
| First slot favoured | Being shown first is worth real odds, and one model takes that slot almost always. | `position_bias`, `position_reordering` |
| Draw-prone model | One weak model draws most of its battles, and half a win each is generous. | `tie_convention` |
| Rock, paper, scissors | Three models beat each other in a circle, which no single strength explains. | `intransitivity` |
| Two separate pools | Two halves of the board that never met, ranked as though they had. | `connectivity` |
| Lopsided budget | The comparison budget is spent mostly on the models at the top. | `coverage_imbalance` |
| A few power voters | Six people with strong tastes cast a third of all votes. | `separation`, `vote_clustering` |
| One loud annotator | Two equally good models, and one annotator with a firm view about them. | `annotator_influence`, `separation`, `vote_clustering` |

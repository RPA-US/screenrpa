from .models import ProcessDiscovery
from django.shortcuts import get_object_or_404
import pygraphviz as pgv
###########################################################################################################################
# case study get phases data  ###########################################################################################
###########################################################################################################################

def get_process_discovery(case_study):
  return get_object_or_404(ProcessDiscovery, case_study=case_study)

def case_study_has_process_discovery(case_study):
  return ProcessDiscovery.objects.filter(case_study=case_study, active=True).exists()

#%%
from json import JSONDecoder
from typing import Optional

###########################################################################################################################
# Trazablity between decision points and rules ############################################################################
###########################################################################################################################

## BRANCH

class Branch:
  """
  A branch is an activity or series of activities and decision points that are reached when the condition of a rule is satisfied
  """
  def __init__(self, label: str, id:str, decision_points: Optional[list['DecisionPoint']] = []):
    self.label = label
    self.id = id
    if not decision_points is None:
      ids: list[str] = list(map(lambda dp: dp.id, decision_points))
      if len(ids) != len(set(ids)):
        raise ValueError("Decision points must have unique ids")
      self.decision_points = decision_points
    
  def to_json(self) -> dict:
    return {
      'label': self.label,
      'id': self.id,
      'decision_points': list(map(lambda dp: dp.to_json(), self.decision_points))
    }
    
  # Define a to string method
  def __str__(self):
    return f"Branch(label={self.label})"
  
  @staticmethod
  def from_json(json: dict) -> 'Branch':
    return BranchDecoder().dict_to_object(json)

class BranchDecoder(JSONDecoder):
  """
  A JSON decoder for Branch objects
  """
  def __init__(self):
    JSONDecoder.__init__(self, object_hook=self.dict_to_object)

  def dict_to_object(self, d: dict) -> Branch:
    # decode decision points into a list of DecisionPoint objects
    decision_points = list(map(lambda dp: DecisionPointDecoder().dict_to_object(dp), d['decision_points']))
    return Branch(d['label'], d['id'], decision_points)

## RULE

class Rule:
  """
  A rule is a condition that is evaluated in a decision point. It has a condition in the form of logic operations and a target branch
  """
  def __init__(self, condition: list[str], target: str):
    self.condition = condition
    self.target = target
    
  # Define a to string method
  def __str__(self):
    return f"Rule(condition={self.condition}, target={self.target})"
  
  @staticmethod
  def from_json(json: dict) -> 'Rule':
    return RuleDecoder().dict_to_object(json)

class RuleDecoder(JSONDecoder):
  """
  A JSON decoder for Rule objects
  """
  def __init__(self):
    JSONDecoder.__init__(self, object_hook=self.dict_to_object)

  def dict_to_object(self, d: dict) -> Rule:
    return Rule(d[list(d.keys())[0]], list(d.keys())[0])

## DECISION POINT

class DecisionPoint:
  """
  A decision point is a point in the process where a decision is made. It has a unique id, a previous activity, a list of branches and a list of rules
  """
  def __init__(self, id: str, prevAct: str, branches: list[Branch], rules: Optional[list[Rule]] = []):
    self.rules = rules
    labels = list(map(lambda b: b.label, branches))
    if len(labels) != len(set(labels)):
      raise ValueError("Branches must have unique labels")
    self.branches = branches
    self.prevAct = prevAct
    self.id = id
  
  def to_json(self) -> dict:
    return {
      'id': self.id,
      'prevAct': self.prevAct,
      'branches': list(map(lambda b: b.to_json(), self.branches)),
      'rules': {
        self.rules[i].target: self.rules[i].condition for i in range(len(self.rules)) 
      }
    }
  
  # Define a to string method
  def __str__(self):
    branches_str = ', '.join(str(b) for b in self.branches)
    return f"DP(id={self.id}, prevAct={self.prevAct}, branches={branches_str})"

  @staticmethod
  def from_json(json: dict) -> 'DecisionPoint':
    return DecisionPointDecoder().dict_to_object(json)
  
class DecisionPointDecoder(JSONDecoder):
  """
  A JSON decoder for DecisionPoint objects
  """
  def __init__(self):
    JSONDecoder.__init__(self, object_hook=self.dict_to_object)

  def dict_to_object(self, d: dict) -> DecisionPoint:
    # decode branches into a list of Branch objects
    branches = list(map(lambda b: BranchDecoder().dict_to_object(b), d['branches']))
    # decode rules into a list of Rule objects
    rules = [RuleDecoder().dict_to_object({k: v}) for k, v in d['rules'].items()]
    return DecisionPoint(d['id'], d['prevAct'], branches, rules)
  
## PROCESS

class Process:
  """
  A processes represented as a sequence of decision points. This is the root of the process
  """
  def __init__(self, decision_points: list[DecisionPoint]):
    ids = list(map(lambda dp: dp.id, decision_points))
    if len(ids) != len(set(ids)):
      raise ValueError("Decision points must have unique ids")
    self.decision_points = decision_points 
   
  def to_json(self) -> dict:
    return {
      'decision_points': list(map(lambda dp: dp.to_json(), self.decision_points))
    }
    
  # Define a to string method
  def __str__(self):
        decision_points_str = ', '.join(str(dp) for dp in self.decision_points)
        return f"Process(decision_points=[{decision_points_str}])"
  
  @staticmethod
  def from_json(json: dict) -> 'Process':
    return ProcessDecoder().dict_to_object(json)

  def get_all_branches_flattened(self) -> list[Branch]:
    """
    Get all branches in the process recursively
    """

    def recursive_search(dp: DecisionPoint):
      branches = []
      for branch in dp.branches:
        branches.append(branch)
        for b in branch.decision_points:
          branches += recursive_search(b)
      return branches
    
    branches = []
    for dp in self.decision_points:
      branches += recursive_search(dp)
    return branches
  
  def get_non_empty_dp_flattened(self) -> list[DecisionPoint]:
    """
    Get all decision points in the process that are not empty
    """

    def recursive_search(branch: Branch):
      dps = []
      for dp in branch.decision_points:
        if dp.branches:
          dps.append(dp)
          for b in dp.branches:
            dps += recursive_search(b)
      return dps

    dps = []
    for dp in self.decision_points:
      if dp.branches:
        dps.append(dp)
      for b in dp.branches:
        dps += recursive_search(b)
    return dps

class ProcessDecoder(JSONDecoder):
  """
  A JSON decoder for Process objects
  """
  def __init__(self):
    JSONDecoder.__init__(self, object_hook=self.dict_to_object)

  def dict_to_object(self, d: dict) -> Process:
    # decode decision points into a list of DecisionPoint objects
    decision_points = list(map(lambda dp: DecisionPointDecoder().dict_to_object(dp), d['decision_points']))
    return Process(decision_points)


# Auxiliary function  
def find_non_empty_decision_points(json_text):
    """
    Find non empty decision points from traceability.json
    
    Args:
        json_text (str): traceability.json as file

    Returns:
        set: non empty decision points
    """
    non_empty_decision_points = set()

    def recursive_search(branch, prevAct):
      if prevAct:
        non_empty_decision_points.add(prevAct)
      if branch['decision_points']:  # Si no está vacío
        for dp in branch['decision_points']:
          for b in dp['branches']:
            recursive_search(b, dp['prevAct'])  # Búsqueda recursiva en los hijos

    # Iniciar la búsqueda desde la raíz
    recursive_search(json_text, None)
    return non_empty_decision_points

##########################################################33


## WORKS UNDER THE ASSUMPTION THAT A DECISION POINT IS ALWAYS REACHED THROUGH AN ACTIVITY (only one predecessor)
# extracts a list with the labels of the predecessors to the decision points
def extract_prev_act_labels(dot_path):
  # Load the graph from a DOT file
  graph = pgv.AGraph(dot_path)
  
  # List to store the labels that precede the decision points with a single predecessor
  unique_predecessor_labels = []
  
  # Identify all nodes that are decision points with label "X"
  decision_points = [node for node in graph.nodes() if graph.get_node(node).attr['label'] == 'X']
  
  # Iterate over each decision point and find the nodes that link to it
  for decision in decision_points:
    # Get the predecessors of the decision point
    predecessors = graph.predecessors(decision)
    # Check that there is only one predecessor
    if len(predecessors) == 1:
      # Get the label of the unique predecessor and add it to the list
      unique_predecessor_labels.append(graph.get_node(predecessors[0]).attr['label'])
  
  return unique_predecessor_labels

def extract_all_activities_labels(dot_path):
    # Load the graph from a DOT file
    graph = pgv.AGraph(dot_path)
    
    # List to store the labels of all activities (boxes)
    activity_labels = []
    
    # Iterate over each node in the graph
    for node in graph.nodes():
        # Check if the node is an activity (box shape) and the label is a number
        if graph.get_node(node).attr['shape'] == 'box' and graph.get_node(node).attr['label'].isdigit():
            # Add the label to the list
            activity_labels.append(int(graph.get_node(node).attr['label']))
    
    return activity_labels
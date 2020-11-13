import random
blocktype = ["gold_block", "emerald_block"]
color = ["WHITE", "MAGENTA", "LIGHT_BLUE", "YELLOW", "LIME", "CYAN", "PURPLE", "BLUE", "BROWN", "GREEN", "RED"]
def GetMissionXML(SIZE,WIDTH):
  myxml = ""
  # bp = 
  bridgeL = int(SIZE/4)
  bridgeW = int(SIZE/8)
  riverW = WIDTH-int(SIZE/8)
  for x in range(-WIDTH,WIDTH+1):
        for y in range(1,int(SIZE/4)):
            # if random.random() < 0.2:
            #     myxml += "<DrawBlock x='{}' y='10' z='{}' type='{}'/>".format(x,y,random.choice(blocktype))
            if random.random() < 0.2:
                myxml += "<DrawCuboid x1='{}' y1='10' z1='{}' x2='{}' y2='12' z2='{}'  type='{}' />".format(x,y,x,y,random.choice(blocktype))
  return '''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
            <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
            
              <About>
                <Summary>Hello world!</Summary>
              </About>
              
            <ServerSection>
              <ServerInitialConditions>
                <Time>
                    <StartTime>7000</StartTime>
                    <AllowPassageOfTime>false</AllowPassageOfTime>
                </Time>
                <Weather>clear</Weather>
              </ServerInitialConditions>
              <ServerHandlers>
                  <FlatWorldGenerator generatorString="3;1;"/>
                  <DrawingDecorator>''' + \
                    "<DrawCuboid x1='{}' y1='0' z1='-6' x2='{}' y2='8' z2='{}' type='stone'/>".format(-WIDTH-1,WIDTH+1,SIZE+1) + \
                    "<DrawCuboid x1='{}' y1='10' z1='-6' x2='{}' y2='15' z2='{}' type='glass'/>".format(-WIDTH-1,WIDTH+1,SIZE+1) + \
                    "<DrawCuboid x1='{}' y1='10' z1='-5' x2='{}' y2='40' z2='{}' type='air'/>".format(-WIDTH,WIDTH,SIZE) + \
                    "<DrawCuboid x1='{}' y1='9' z1='-6' x2='{}' y2='9' z2='-1' type='wool' colour='PINK'/>".format(-WIDTH-1,WIDTH+1) + \
                    "<DrawCuboid x1='{}' y1='9' z1='0' x2='{}' y2='9' z2='{}' type='wool' colour='{}'/>".format(-WIDTH-1,WIDTH+1,int(SIZE/4),random.choice(color)) + \
                    "<DrawCuboid x1='{}' y1='9' z1='{}' x2='{}' y2='9' z2='{}' type='diamond_block'/>".format(-WIDTH-1,int(SIZE/4)+1,WIDTH+1,int(SIZE/4)+1) + \
                    "<DrawCuboid x1='{}' y1='9' z1='{}' x2='{}' y2='9' z2='{}' type='wool' colour='{}'/>".format(-WIDTH-1,int(SIZE/4)+2,WIDTH+1,SIZE+1,random.choice(color)) + \
                    "<DrawCuboid x1='{}' y1='10' z1='{}' x2='{}' y2='20' z2='{}' type='redstone_block'/>".format(-WIDTH,SIZE,WIDTH,SIZE) + \
                    "<DrawCuboid x1='{}' y1='8' z1='-5' x2='{}' y2='8' z2='{}' type='water'/>".format(-WIDTH,WIDTH,SIZE) + \
                    myxml + \
                  '''</DrawingDecorator>
                  <AnimationDecorator ticksPerUpdate="40">
                    <Linear>
                      <CanvasBounds>''' +\
                        "<min x='{}' y='9' z='0'/>".format(-WIDTH-riverW-1) +\
                        "<max x='{}' y='9' z='100'/>".format(WIDTH+riverW) +\
                      '''</CanvasBounds>'''+ \
                      "<InitialPos x='{}' y='9' z='{}'/>".format(-WIDTH-riverW-1,int(SIZE/3)) +\
                      "<InitialVelocity x='1' y='0' z='0'/>" +\
                    '''</Linear>
                    <DrawingDecorator>''' + \
                      "<DrawCuboid x1='0' y1='0' z1='0'  x2='{}' y2='0' z2='{}' type='glass'/>".format(riverW-1,bridgeL) +\
                      "<DrawCuboid x1='{}' y1='0' z1='0'  x2='{}' y2='0' z2='{}' type='wool' colour='ORANGE'/>".format(riverW,riverW+bridgeW,bridgeL) +\
                      "<DrawCuboid x1='{}' y1='0' z1='0'  x2='{}' y2='0' z2='{}' type='glass'/>".format(riverW+bridgeW+1,2*riverW+bridgeW-1,bridgeL) +\
                      "<DrawCuboid x1='{}' y1='0' z1='0'  x2='{}' y2='0' z2='{}' type='wool' colour='ORANGE'/>".format(2*riverW+bridgeW,2*riverW+2*bridgeW,bridgeL) +\
                      "<DrawCuboid x1='{}' y1='0' z1='0'  x2='{}' y2='0' z2='{}' type='glass'/>".format(2*riverW+2*bridgeW+1,3*riverW+2*bridgeW,bridgeL) +\
                    '''</DrawingDecorator>
                  </AnimationDecorator>
                  <ServerQuitFromTimeUp timeLimitMs="300000"/>
                  <ServerQuitWhenAnyAgentFinishes/>
                </ServerHandlers>
              </ServerSection>
              
              <AgentSection mode="Survival">
                <Name>Diablo!</Name>
                <AgentStart>
                    <Placement x="0.5" y="10" z="0.5" yaw="0"/>
                </AgentStart>
                <AgentHandlers>
                  <DiscreteMovementCommands/>
                  <ObservationFromFullStats/>
                </AgentHandlers>
              </AgentSection>
            </Mission>'''